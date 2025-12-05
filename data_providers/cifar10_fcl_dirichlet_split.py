# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import os

from data_providers.base_provider import *
from prefetch_generator import BackgroundGenerator

from typing import List, Tuple, Dict, Sequence
import numpy as np
import random
import json
from datetime import datetime


class FCLDataManagerC10(object):
    """
    面向类的 FCL 数据划分管理器（CIFAR-10，10 类）。
    - 按任务（每任务若干类）+ 客户端（Dirichlet 非IID 或 IID）生成训练索引，并从训练中再切分 10% 为验证；
      测试集按“累计类别”生成（所有客户端共享同一 test_set）。
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
        # 允许外部仅传 num_tasks，由这里推导 classes_per_task
        if classes_per_task is None:
            if num_tasks <= 0 or 10 % num_tasks != 0:
                raise ValueError(f"CIFAR-10 无法将 10 个类别均分到 {num_tasks} 个任务，请检查 num_tasks")
            classes_per_task = 10 // num_tasks
        assert num_tasks * classes_per_task == 10, "CIFAR-10：num_tasks × classes_per_task 必须等于 10"
        self.train_targets = np.asarray(train_targets)
        self.test_targets = np.asarray(test_targets)
        self.K = int(num_clients)
        self.T = int(num_tasks)
        self.CPT = int(classes_per_task)
        self.alpha = float(alpha)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.iid = bool(iid)

        self._perm_classes: List[int] = self._build_perm_classes(self.seed)  # 0..9 的打乱
        self._task_classes: Dict[int, List[int]] = {t: self._classes_of(t) for t in range(1, self.T + 1)}
        self._cumu_classes: Dict[int, List[int]] = {t: self._classes_upto(t) for t in range(1, self.T + 1)}

        self._train_valid_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._test_cache: Dict[int, np.ndarray] = {}

        if precompute:
            self.precompute_all()

    def _build_perm_classes(self, seed: int) -> List[int]:
        rng = random.Random(seed)
        classes = list(range(10))
        rng.shuffle(classes)
        return classes

    def _classes_of(self, task_id: int) -> List[int]:
        assert 1 <= task_id <= self.T
        st = (task_id - 1) * self.CPT
        ed = st + self.CPT
        return self._perm_classes[st:ed]

    def _classes_upto(self, task_id: int) -> List[int]:
        assert 1 <= task_id <= self.T
        return self._perm_classes[: task_id * self.CPT]

    @staticmethod
    def _filter_indices_by_classes(targets: np.ndarray, classes: Sequence[int]) -> np.ndarray:
        idx_all = np.arange(len(targets))
        cls_set = set(int(c) for c in classes)
        mask = np.vectorize(lambda y: int(y) in cls_set)(targets)
        return idx_all[mask]

    @staticmethod
    def _group_indices_by_class(targets: np.ndarray, indices: np.ndarray, classes: Sequence[int],
                                shuffle_seed: int) -> Dict[int, np.ndarray]:
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

    def _build_task_train_valid(self, task_id: int):
        task_classes = self._task_classes[task_id]
        pool_indices = self._filter_indices_by_classes(self.train_targets, task_classes)
        class2idx = self._group_indices_by_class(self.train_targets, pool_indices, task_classes,
                                                 shuffle_seed=2024 + task_id)

        rng = np.random.RandomState(self.seed + 1000 + task_id)
        client_buckets: Dict[int, list] = {k: [] for k in range(self.K)}

        for c in task_classes:
            idx_c = class2idx[c]
            n = len(idx_c)
            if n == 0:
                continue
            if self.iid:
                base = n // self.K
                rem = n % self.K
                counts = np.full(self.K, base, dtype=int)
                counts[:rem] += 1
            else:
                p = rng.dirichlet(alpha=np.full(self.K, self.alpha))
                counts = (p * n).astype(int)
                while counts.sum() < n:
                    counts[np.argmax(p)] += 1
                while counts.sum() > n:
                    counts[np.argmin(p)] -= 1
            st = 0
            for k in range(self.K):
                take = counts[k]
                if take > 0:
                    client_buckets[k].extend(idx_c[st:st + take].tolist())
                    st += take

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

    def _build_task_test(self, task_id: int):
        classes = self._cumu_classes[task_id]
        tst_idx = self._filter_indices_by_classes(self.test_targets, classes)
        self._test_cache[task_id] = tst_idx

    def get(self, client_id: int, task_id: int) -> Tuple[List[int], List[int], List[int]]:
        assert 0 <= client_id < self.K, "client_id 越界"
        assert 1 <= task_id <= self.T, "task_id 越界"

        if (task_id, client_id) not in self._train_valid_cache:
            self._build_task_train_valid(task_id)
        if task_id not in self._test_cache:
            self._build_task_test(task_id)

        trn, val = self._train_valid_cache[(task_id, client_id)]
        tst = self._test_cache[task_id]
        return trn.tolist(), val.tolist(), tst.tolist()

    def precompute_all(self):
        for t in range(1, self.T + 1):
            self._build_task_train_valid(t)
            self._build_task_test(t)

    def save_partitions(self, out_file: str):
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

        with open(out_file, "w") as fout:
            json.dump(out, fout, indent=2)
        return out_file


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CifarDataProvider10(DataProvider):
    def __init__(self, client_id=10, dataset_location=None, train_batch_size=1024,
                 test_batch_size=256, n_worker=2,
                 num_clients=10, num_tasks=10, classes_per_task=1, alpha=0.3, val_ratio=0.1, seed=0,
                 search=True, task_id=1, is_client=True, iid=False):
        self.client_id = client_id
        self.task_id = task_id
        self.dset_save_path = dataset_location
        self.search = search

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        dataset_class = datasets.CIFAR10

        train_transforms, test_transform = self.cifar_data_transforms()

        self.train_dataset = dataset_class(root=self.train_path,
                                           train=True,
                                           download=False,
                                           transform=train_transforms)
        self.test_dataset = dataset_class(root=self.valid_path,
                                          train=False,
                                          download=False,
                                          transform=test_transform)

        precompute = client_id == 0 and task_id == 1 and is_client

        self.fcl_manager = FCLDataManagerC10(
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
                save_path = os.path.join(self.dset_save_path, f"fcl_partitions_c10_{timestamp}.json")
                self.fcl_manager.save_partitions(save_path)
            except Exception:
                pass

        trn_idx, val_idx, tst_idx = self.fcl_manager.get(client_id=self.client_id, task_id=self.task_id)

        self.trn_set = Subset(self.train_dataset, trn_idx)
        self.val_set = Subset(self.train_dataset, val_idx)
        self.test_dataset = Subset(self.test_dataset, tst_idx)

        self.trn_set_length = len(self.trn_set)
        self.val_set_length = len(self.val_set)
        self.tst_set_length = len(self.test_dataset)

        self.n_worker = n_worker

    @staticmethod
    def name():
        return 'cifar10'

    @property
    def data_shape(self):
        return 3, self.image_size, self.image_size

    @property
    def n_classes(self):
        return 10

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
        if self.search:
            print("搜索阶段使用验证集进行精度评估")
            return DataLoaderX(self.val_set,
                               batch_size=self.test_batch_size,
                               shuffle=True,
                               num_workers=self.n_worker,
                               pin_memory=True,
                               drop_last=False)
        else:
            print("重训阶段使用测试集进行精度评估")
            return DataLoaderX(self.test_dataset,
                               batch_size=self.test_batch_size,
                               shuffle=True,
                               num_workers=self.n_worker,
                               pin_memory=True,
                               drop_last=False)

    def test(self):
        return DataLoaderX(self.test_dataset,
                           batch_size=self.test_batch_size,
                           num_workers=self.n_worker,
                           pin_memory=True,
                           drop_last=False)


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
