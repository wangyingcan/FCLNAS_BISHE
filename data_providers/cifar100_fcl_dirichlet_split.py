# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from data_providers.base_provider import *
from prefetch_generator import BackgroundGenerator

from typing import List, Tuple, Dict, Sequence
import numpy as np
import random
import json
from datetime import datetime

class FCLDataManager(object):
    """
    面向类的 FCL 数据划分管理器。
    - 数据集：默认针对 CIFAR-100（100 类），可通过 tasks × classes_per_task 自由配置（需相乘为 100）。
    - 划分：按任务（每任务若干类）+ 客户端（Dirichlet 非IID）生成训练索引，并从训练中再切分 10% 为验证；
            测试集按“累计类别”生成（所有客户端共享同一 test_set）。
    - 缓存：内部以 (task_id, client_id) 作为键缓存 train/valid，按 task_id 缓存 test_set。
    """
    def __init__(self,
                 train_targets: Sequence[int],
                 test_targets: Sequence[int],
                 num_clients: int = 10,
                 num_tasks: int = 10,
                 classes_per_task: int = None,
                 alpha: float = 0.3,
                 val_ratio: float = 0.1,
                 seed: int = 0,
                 precompute: bool = True,
                 iid: bool = False):
        # ---------- 基础参数 ----------
        if classes_per_task is None:
            if num_tasks <= 0 or 100 % num_tasks != 0:
                raise ValueError(f"CIFAR-100 无法将 100 个类别均分到 {num_tasks} 个任务，请检查 num_tasks")
            classes_per_task = 100 // num_tasks
        assert num_tasks * classes_per_task == 100, "CIFAR-100：num_tasks × classes_per_task 必须等于 100"
        self.train_targets = np.asarray(train_targets)
        self.test_targets = np.asarray(test_targets)
        self.K = int(num_clients)
        self.T = int(num_tasks)
        self.CPT = int(classes_per_task)
        self.alpha = float(alpha)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.iid = bool(iid)

        # ---------- 任务类别规划 ----------
        self._perm_classes: List[int] = self._build_perm_classes(self.seed)  # 0..99 的打乱
        # 每个任务的类别列表（1-based 任务编号）
        self._task_classes: Dict[int, List[int]] = {
            t: self._classes_of(t) for t in range(1, self.T + 1)
        }
        # 每个任务的“累计类别列表”（用于累计测试集）
        self._cumu_classes: Dict[int, List[int]] = {
            t: self._classes_upto(t) for t in range(1, self.T + 1)
        }

        # ---------- 缓存结构 ----------
        # 训练/验证：key=(t, k) -> (train_idx, valid_idx)
        self._train_valid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        # 测试：key=t -> test_idx
        self._test_cache: Dict[int, np.ndarray] = {}

        # （可选）预先生成全部划分
        if precompute:
            self.precompute_all()

    # ================= 工具：任务与类别 =================
    def _build_perm_classes(self, seed: int) -> List[int]:
        """按 seed 打乱 0..99 类别，返回置换序列。"""
        rng = random.Random(seed)
        classes = list(range(100))
        rng.shuffle(classes)
        return classes

    def _classes_of(self, task_id: int) -> List[int]:
        """返回 task_id 的类别列表（当前任务的 CPT 个类）。"""
        assert 1 <= task_id <= self.T
        st = (task_id - 1) * self.CPT
        ed = st + self.CPT
        return self._perm_classes[st:ed]

    def _classes_upto(self, task_id: int) -> List[int]:
        """返回从任务1累计到 task_id 的所有类别列表（用于累计测试）。"""
        assert 1 <= task_id <= self.T
        return self._perm_classes[: task_id * self.CPT]

    # ================= 工具：索引筛选与分桶 =================
    @staticmethod
    def _filter_indices_by_classes(targets: np.ndarray, classes: Sequence[int]) -> np.ndarray:
        """筛选 targets 中属于 classes 的全局索引。"""
        idx_all = np.arange(len(targets))
        cls_set = set(int(c) for c in classes)
        mask = np.vectorize(lambda y: int(y) in cls_set)(targets)
        return idx_all[mask]

    @staticmethod
    def _group_indices_by_class(targets: np.ndarray, indices: np.ndarray, classes: Sequence[int],
                                shuffle_seed: int) -> Dict[int, np.ndarray]:
        """
        将给定的全局索引 indices，按类别映射到 {class_id -> ndarray(indices)}，
        并对每类内部做一次固定 seed 的洗牌。
        """
        buckets: Dict[int, list] = {int(c): [] for c in classes}
        for gidx in indices:
            c = int(targets[gidx])
            if c in buckets:
                buckets[c].append(int(gidx))
        out: Dict[int, np.ndarray] = {}
        for c, lst in buckets.items():
            arr = np.asarray(lst, dtype=int)
            np.random.RandomState(shuffle_seed + c).shuffle(arr)
            out[int(c)] = arr
        return out

    # ================= 生成：单个任务的客户端划分 =================
    def _build_task_train_valid(self, task_id: int):
        """
        为 task_id 构建所有客户端的训练/验证索引，并写入 _train_valid_cache。
        - 训练样本池：仅当前任务的 CPT 个类（不累计）；
        - 客户端划分：对任务池内的每个类别，按 Dirichlet(α) 将该类样本分给 K 个客户端（非重叠）；
        - 本地 train/valid：对客户端所分得的样本，按 val_ratio 再切分 10%。
        """
        # 任务内训练样本池（仅当前任务的类）
        task_classes = self._task_classes[task_id]
        pool_indices = self._filter_indices_by_classes(self.train_targets, task_classes)
        # 类内分桶（加速后续切片）
        class2idx = self._group_indices_by_class(self.train_targets, pool_indices, task_classes,
                                                 shuffle_seed=2024 + task_id)

        # 逐类做 Dirichlet 到 K 客户端的分配
        rng = np.random.RandomState(self.seed + 1000 + task_id)
        # 先为每个客户端准备收集容器
        client_buckets: Dict[int, list] = {k: [] for k in range(self.K)}

        for c in task_classes:
            idx_c = class2idx[c]
            n = len(idx_c)
            if n == 0:
                continue
            if self.iid:
                # 均匀分到每个客户端（IID），余数按轮转分配
                base = n // self.K
                rem = n % self.K
                counts = np.full(self.K, base, dtype=int)
                counts[:rem] += 1
            else:
                # 非IID：按 Dirichlet(α) 采样比例
                p = rng.dirichlet(alpha=np.full(self.K, self.alpha))
                counts = (p * n).astype(int)
                # 修正四舍五入导致的和不等于 n 的情况
                while counts.sum() < n:
                    counts[np.argmax(p)] += 1
                while counts.sum() > n:
                    counts[np.argmin(p)] -= 1
            # 切片分配（不重叠）
            st = 0
            for k in range(self.K):
                take = counts[k]
                if take > 0:
                    client_buckets[k].extend(idx_c[st:st + take].tolist())
                    st += take

        # 对每个客户端，打乱并切分 train/valid
        for k in range(self.K):
            arr = np.asarray(client_buckets[k], dtype=int)
            rng.shuffle(arr)
            if len(arr) == 0:
                trn = arr
                val = np.asarray([], dtype=int)
            else:
                n_val = int(round(len(arr) * self.val_ratio))
                perm = rng.permutation(len(arr))
                val_sel = perm[:n_val]
                trn_sel = perm[n_val:]
                trn = arr[trn_sel]
                val = arr[val_sel]
            self._train_valid_cache[(task_id, k)] = (trn, val)

    # ================= 生成：单个任务的累计测试索引 =================
    def _build_task_test(self, task_id: int):
        """为 task_id 构建累计测试索引，并写入 _test_cache。"""
        classes = self._cumu_classes[task_id]
        tst_idx = self._filter_indices_by_classes(self.test_targets, classes)
        self._test_cache[task_id] = tst_idx

    # ================= 公共：获取接口（惰性生成 + 缓存） =================
    def get(self, client_id: int, task_id: int) -> Tuple[List[int], List[int], List[int]]:
        """
        返回 (train_idx, valid_idx, test_idx) 三元组（均为全局索引 list）。
        若未生成对应划分，则先生成再返回。
        """
        assert 0 <= client_id < self.K, "client_id 越界"
        assert 1 <= task_id <= self.T, "task_id 越界"

        if (task_id, client_id) not in self._train_valid_cache:
            self._build_task_train_valid(task_id)
        if task_id not in self._test_cache:
            self._build_task_test(task_id)

        trn, val = self._train_valid_cache[(task_id, client_id)]
        tst = self._test_cache[task_id]
        return trn.tolist(), val.tolist(), tst.tolist()

    # ================= 批量预生成（可选） =================
    def precompute_all(self):
        """一次性生成所有任务 × 客户端的 train/valid 与每个任务的 test。"""
        for t in range(1, self.T + 1):
            self._build_task_train_valid(t)
            self._build_task_test(t)

    # ================= 可选：导出摘要（便于调试/记录） =================
    def summary(self) -> Dict[str, dict]:
        """
        返回一个字典摘要，包含每个任务/客户端的数据量统计，便于检查划分是否合理。
        """
        info: Dict[str, dict] = {"tasks": {}, "config": {}}
        for t in range(1, self.T + 1):
            task_info = {"clients": {}, "test_size": int(self._test_cache.get(t, np.array([])).size)}
            for k in range(self.K):
                trn, val = self._train_valid_cache.get((t, k), (np.array([]), np.array([])))
                task_info["clients"][k] = {"train": int(trn.size), "valid": int(val.size)}
            info["tasks"][t] = task_info
        info["config"] = {
            "num_clients": self.K,
            "num_tasks": self.T,
            "classes_per_task": self.CPT,
            "alpha": self.alpha,
            "val_ratio": self.val_ratio,
            "seed": self.seed,
        }
        return info
    
    def save_partitions(self, out_file: str):
        """
        将当前已生成的 train/valid/test 划分信息写为 JSON。
        包含：每个 task 的 classes、每个 client 的 train/val 索引与样本数、每个 client 的训练类集合、以及全局配置信息。
        """
        out = {"config": {"num_clients": self.K, "num_tasks": self.T, "classes_per_task": self.CPT,
                          "alpha": self.alpha, "val_ratio": self.val_ratio, "seed": self.seed},
               "tasks": {}}
        for t in range(1, self.T + 1):
            task_entry = {}
            task_entry["classes"] = list(self._task_classes.get(t, []))
            tst_idx = self._test_cache.get(t, np.array([], dtype=int))
            task_entry["test_indices"] = tst_idx.tolist() if hasattr(tst_idx, "tolist") else list(tst_idx)
            clients = {}
            for k in range(self.K):
                trn, val = self._train_valid_cache.get((t, k), (np.array([], dtype=int), np.array([], dtype=int)))
                trn_list = trn.tolist() if hasattr(trn, "tolist") else list(trn)
                val_list = val.tolist() if hasattr(val, "tolist") else list(val)
                # 从全局 train_targets 推断该 client 在该任务上的类别集合
                if len(trn_list) > 0:
                    train_classes = sorted(list({int(self.train_targets[idx]) for idx in trn_list}))
                else:
                    train_classes = []
                clients[k] = {
                    "train_count": int(len(trn_list)),
                    "val_count": int(len(val_list)),
                    "train_indices": trn_list,
                    "val_indices": val_list,
                    "train_classes": train_classes
                }
            task_entry["clients"] = clients
            out["tasks"][str(t)] = task_entry

        # 写文件
        with open(out_file, "w") as fout:
            json.dump(out, fout, indent=2)
        return out_file

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CifarDataProvider100(DataProvider):
    def __init__(self, client_id=10, dataset_location=None, train_batch_size=1024,
                 test_batch_size=256, n_worker=2,
                 num_clients=10, num_tasks=10, classes_per_task=10, alpha=0.3, val_ratio=0.1, seed=0,
                 search=True, task_id=1, is_client=True, iid=False):
        self.client_id = client_id
        self.task_id = task_id
        self.dset_save_path = dataset_location
        self.search = search

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        dataset_class = datasets.CIFAR100

        train_transforms, test_transform = self.cifar_data_transforms()

        self.train_dataset = dataset_class(root=self.train_path,
                                           train=True,
                                           download=False,
                                           transform=train_transforms)
        self.test_dataset = dataset_class(root=self.valid_path,
                                          train=False,
                                          download=False,
                                          transform=test_transform)
        
        if(client_id == 0 and task_id == 1 and is_client):
            precompute = True
        else:
            precompute = False
        
        # 初始化 FCL 数据管理器
        self.fcl_manager = FCLDataManager(
            train_targets=self.train_dataset.targets,
            test_targets=self.test_dataset.targets,
            num_clients=num_clients,
            num_tasks=num_tasks,
            classes_per_task=classes_per_task,
            alpha=alpha,
            val_ratio=val_ratio,
            seed=seed,
            precompute=precompute,
            iid=iid
        )
        
        if precompute:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.dset_save_path, f"fcl_partitions_{timestamp}.json")
                self.fcl_manager.save_partitions(save_path)
            except Exception:
                # 若无法保存（权限/路径等），忽略但不影响训练
                pass
        
        # 划分修改
        trn_idx, val_idx, tst_idx = self.fcl_manager.get(client_id=self.client_id, task_id=self.task_id)

        # 根据索引创建数据子集
        self.trn_set = Subset(self.train_dataset, trn_idx)
        self.val_set = Subset(self.train_dataset, val_idx)
        self.test_dataset = Subset(self.test_dataset, tst_idx)

        # 数据集长度
        self.trn_set_length = len(self.trn_set)
        self.val_set_length = len(self.val_set)
        self.tst_set_length = len(self.test_dataset)
        
        self.n_worker = n_worker

    @staticmethod
    def name():
        return 'cifar100'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size  # C, H, W

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self.dset_save_path is None:
            raise ValueError('unable to Use cifar10')
        return self.dset_save_path

    @property
    def data_url(self):
        raise ValueError('unable to Use cifar10')

    @property
    def train_path(self):
        return self.dset_save_path

    @property
    def valid_path(self):
        return self.dset_save_path

    def cifar_data_transforms(self, cutout_length=8):
        # https://github.com/uoguelph-mlrg/Cutout
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.2023, 0.1994, 0.2010]
        transf = [
            transforms.RandomCrop(32, padding=3),
            transforms.RandomHorizontalFlip()
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]
        train_transform = transforms.Compose(transf + normalize)

        if cutout_length > 0:
            train_transform.transforms.append(Cutout(cutout_length))

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        return train_transform, test_transform

    @property
    def resize_value(self):
        return 36

    @property
    def image_size(self):
        return 32

    def train(self):
        return DataLoaderX(
            self.trn_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.n_worker,
            pin_memory=True,
            drop_last=False,
        )

    def valid(self):
        # valid调用时对于search阶段使用验证集
        if(self.search):
            print("搜索阶段使用验证集进行精度评估")
            return DataLoaderX(self.val_set,
                           batch_size=self.test_batch_size,
                           shuffle=True,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )
        # valid调用时对于retrain阶段使用测试集
        else:
            print("重训阶段使用测试集进行精度评估")
            return DataLoaderX(self.test_dataset,
                           batch_size=self.test_batch_size,
                           shuffle=True,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )
            

    def test(self):
        return DataLoaderX(self.test_dataset,
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False,
                           )
        
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img
