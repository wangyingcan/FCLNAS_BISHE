import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# 确保工程根目录在 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import get_net_by_name  # noqa: E402
from models.baseline_nets import BaselineResNet  # noqa: E402
from run_manager import RunManager, CifarRunConfig, SimpleReplayBuffer  # noqa: E402

try:
    from tensorboardX import SummaryWriter
except Exception:
    SummaryWriter = None


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        return

    def close(self):
        return


def _build_writer(log_dir: str):
    if SummaryWriter is None:
        return _NullWriter()
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(logdir=log_dir)


# ---------- 模型构建 ----------
def build_model(name: str, num_classes: int, proxyless_config: Optional[str] = None) -> nn.Module:
    name = name.lower()
    if name == "proxyless_subnet":
        # 默认从环境变量或参数指定的 config 加载子网
        config_path = proxyless_config or os.environ.get("PROXYLESS_SUBNET_CONFIG", None)
        if not config_path:
            raise ValueError("未指定 proxyless_subnet 的 config，请通过 --proxyless_config 或环境变量 PROXYLESS_SUBNET_CONFIG 提供")
        if config_path.endswith((".pt", ".pth")):
            cfg = torch.load(config_path, map_location="cpu")
        else:
            with open(config_path, "r") as fin:
                cfg = json.load(fin)
        net = get_net_by_name("ProxylessNASNets").build_from_config(cfg)
        return net
    if name == "resnet18":
        return BaselineResNet(arch="resnet18", num_classes=num_classes, pretrained=False)
    if name == "small_model":
        # 以 torchvision mobilenet_v2(0.5) 作为小模型
        model = models.mobilenet_v2(width_mult=0.5, num_classes=num_classes)
        # 补齐 RunManager 期望的接口
        if not hasattr(model, "init_model"):
            model.init_model = lambda model_init=None, init_div_groups=False: None
        if not hasattr(model, "set_bn_param"):
            model.set_bn_param = lambda momentum, eps: None
        if not hasattr(model, "get_flops"):
            model.get_flops = lambda x: (0, None)
        if not hasattr(model, "module_str"):
            model.module_str = "mobilenet_v2_0.5"
        if not hasattr(model, "config"):
            model.config = {"name": "mobilenet_v2_0.5", "num_classes": num_classes}
        if not hasattr(model, "get_parameters"):
            def _get_parameters(keys=None, mode="exclude"):
                keys = keys or []
                if isinstance(keys, str):
                    keys = keys.split("#")
                params = []
                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    is_nd = any(k in name for k in keys)
                    if (mode == "include" and is_nd) or (mode != "include" and not is_nd):
                        params.append(p)
                return params
            model.get_parameters = _get_parameters
        if not hasattr(model, "weight_parameters"):
            model.weight_parameters = lambda: [p for p in model.parameters() if p.requires_grad]
        return model
    raise ValueError(f"未知 backbone: {name}")


# ---------- 数据与任务划分 ----------
def build_task_indices(num_classes: int, num_tasks: int, classes_per_task: int, seed: int = 0) -> List[List[int]]:
    rng = random.Random(seed)
    classes = list(range(num_classes))
    rng.shuffle(classes)
    tasks = []
    for t in range(num_tasks):
        st, ed = t * classes_per_task, (t + 1) * classes_per_task
        tasks.append(classes[st:ed])
    return tasks


def make_dataloaders(dataset_root: str, task_classes: List[int], train_batch: int, test_batch: int, num_workers: int, dataset: str):
    transform_trn = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262])]
    )
    transform_tst = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262])]
    )
    if dataset.lower() == "cifar10":
        train_set = datasets.CIFAR10(root=dataset_root, train=True, download=False, transform=transform_trn)
        test_set = datasets.CIFAR10(root=dataset_root, train=False, download=False, transform=transform_tst)
    else:
        train_set = datasets.CIFAR100(root=dataset_root, train=True, download=False, transform=transform_trn)
        test_set = datasets.CIFAR100(root=dataset_root, train=False, download=False, transform=transform_tst)

    def _filter(ds, classes):
        idx = [i for i, y in enumerate(ds.targets) if y in classes]
        return Subset(ds, idx)

    trn_subset = _filter(train_set, task_classes)
    tst_subset = _filter(test_set, task_classes)
    train_loader = DataLoader(trn_subset, batch_size=train_batch, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(tst_subset, batch_size=test_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------- 简易单客户端训练 ----------
def reset_run_manager_task(run_mgr: RunManager, task_id: int):
    run_mgr.task_id = task_id
    run_mgr.run_config.task_id = task_id
    dp = getattr(run_mgr.run_config, "_data_provider", None)
    if dp is not None:
        dp.task_id = task_id
    run_mgr.run_config._data_provider = None
    run_mgr.run_config._train_iter = None
    run_mgr.run_config._valid_iter = None
    run_mgr.run_config._test_iter = None


def _attach_replay_cfg(run_cfg, args):
    """将回放相关超参写入 run_config，便于 RunManager 读取。"""
    for k in [
        "replay_mode",
        "replay_capacity",
        "replay_per_batch",
        "replay_old_task_scale",
        "replay_old_task_scale_by_F",
        "replay_priority_mode",
    ]:
        if hasattr(args, k):
            setattr(run_cfg, k, getattr(args, k))


def train_single_client(backbone_name: str, args, device: torch.device, save_dir: str, tasks: List[List[int]]) -> str:
    num_classes = 10 if args.dataset.lower() == "cifar10" else 100
    model = build_model(backbone_name, num_classes=num_classes, proxyless_config=args.proxyless_config)
    model = model.to(device)

    run_cfg = CifarRunConfig(
        client_id=0,
        dataset_location=args.dataset_location,
        n_epochs=args.local_epochs,
        init_lr=args.init_lr,
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        n_worker=args.n_worker,
        search=False,
        task_id=1,
        is_client=True,
        num_tasks=args.num_tasks,
        num_users=1,
        classes_per_task=args.classes_per_task,
        ewc_lambda=args.ewc_lambda,
        ewc_samples_per_task=args.ewc_samples_per_task,
        ewc_online_interval=args.ewc_online_interval,
        cl_reg_method=args.cl_reg_method,
        cl_reg_decay=args.cl_reg_decay,
        cl_reg_clip=args.cl_reg_clip,
        cl_penalty_clip=args.cl_penalty_clip,
        cl_kd_method=args.cl_kd_method,
        cl_kd_logit_lambda=args.cl_kd_logit_lambda,
        cl_kd_temperature=args.cl_kd_temperature,
        cl_kd_conf_threshold=args.cl_kd_conf_threshold,
        cl_ortho_method=args.cl_ortho_method,
        cl_ortho_scale=args.cl_ortho_scale,
        ortho_samples_per_task=args.ortho_samples_per_task,
        replay_mode=args.replay_mode,
        replay_capacity=args.replay_capacity,
        replay_per_batch=args.replay_per_batch,
        replay_old_task_scale=args.replay_old_task_scale,
        replay_old_task_scale_by_F=args.replay_old_task_scale_by_F,
        replay_priority_mode=args.replay_priority_mode,
    )
    _attach_replay_cfg(run_cfg, args)
    run_path = os.path.join(save_dir, f"cil_{backbone_name}")
    rm = RunManager(
        path=run_path,
        net=model,
        run_config=run_cfg,
        out_log=True,
        task_id=1,
        measure_latency=None,
        init_model=True,
        replay_buffer=SimpleReplayBuffer(args.replay_capacity),
    )
    writer = _build_writer(os.path.join(run_path, "tensorboard"))
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    teacher_state = None
    for task_id, cls in enumerate(tasks, start=1):
        reset_run_manager_task(rm, task_id)
        print(f"[CIL] backbone={backbone_name} task={task_id} classes={cls}")
        # 若上一任务已有 teacher，设置到当前 RunManager 以启用 KD/ortho
        if teacher_state is not None:
            teacher = build_model(backbone_name, num_classes=num_classes, proxyless_config=args.proxyless_config)
            teacher.load_state_dict(teacher_state, strict=False)
            rm.set_teacher(teacher)
        else:
            rm.set_teacher(None)

        trn_loss, trn_top1, trn_top5, val_loss, val_top1, val_top5, lr = rm.train_run_manager(
            start_local_epoch=0,
            last_local_epoch=args.local_epochs,
            print_top5=True,
            server_model=None,
            writer=writer,
            global_round_idx=task_id - 1,
        )
        print(
            f"[CIL] task{task_id} loss={trn_loss:.4f} top1={trn_top1:.2f} val_top1={val_top1}"
        )
        # 每个任务结束后评估一次（累积测试集）
        rm.validate(is_test=True)

        # 估计 Fisher/ortho 参考，供下一任务遗忘正则
        if args.ewc_lambda > 0:
            fisher, processed = rm.compute_importance(max_samples=args.ewc_samples_per_task)
            if fisher is not None:
                rm.consolidate_ewc(fisher, update_prev_params=True)
                print(f"[EWC] task{task_id} fisher_keys={len(fisher)}, processed={processed}")
        if args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
            ortho_ref, processed = rm.compute_ortho_reference(max_samples=args.ortho_samples_per_task)
            if ortho_ref is not None:
                print(f"[Ortho] task{task_id} ref_keys={len(ortho_ref)}, processed={processed}")

        # 保存教师状态以供下一任务 KD
        teacher_state = copy.deepcopy(rm.net.module.state_dict() if isinstance(rm.net, torch.nn.DataParallel) else rm.net.state_dict())

    ckpt_path = os.path.join(ckpt_dir, f"cil_{backbone_name}_final.pth")
    torch.save({"state_dict": rm.net.module.state_dict() if isinstance(rm.net, torch.nn.DataParallel) else rm.net.state_dict()}, ckpt_path)
    writer.close()
    return ckpt_path


# ---------- 特征抽取 ----------
def _penultimate_hook(model: nn.Module, backbone_name: str):
    outputs = {}

    def hook_fn(_m, _inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        outputs["feat"] = out.detach()

    name = backbone_name.lower()
    if name == "proxyless_subnet":
        handle = model.global_avg_pooling.register_forward_hook(hook_fn)
    elif name == "resnet18":
        handle = model.backbone.avgpool.register_forward_hook(hook_fn)
    else:  # small_model
        # mobilenet_v2: use avgpool
        target = model.avgpool if hasattr(model, "avgpool") else model.features[-1]
        handle = target.register_forward_hook(hook_fn)
    return outputs, handle


def extract_features_for_backbone(backbone_name: str, checkpoint_path: str, feature_save_path: str,
                                  dataset_root: str, tasks: List[List[int]], samples_per_class: int = 50,
                                  train_batch_size: int = 128, num_workers: int = 2,
                                  proxyless_config: Optional[str] = None, dataset: str = "CIFAR100"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if dataset.lower() == "cifar10" else 100
    model = build_model(backbone_name, num_classes=num_classes, proxyless_config=proxyless_config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.491, 0.482, 0.447], [0.247, 0.243, 0.262])]
    )
    test_set = datasets.CIFAR100(root=dataset_root, train=False, download=False, transform=transform)

    feats, labels, task_ids = [], [], []
    outputs, handle = _penultimate_hook(model, backbone_name)
    try:
        for task_id, cls_list in enumerate(tasks, start=1):
            # 为该任务每类采样 samples_per_class
            cls_indices: Dict[int, List[int]] = {c: [] for c in cls_list}
            for idx, y in enumerate(test_set.targets):
                if y in cls_indices:
                    cls_indices[y].append(idx)
            selected_idx = []
            for c, idxs in cls_indices.items():
                random.shuffle(idxs)
                selected_idx.extend(idxs[:samples_per_class])
            subset = Subset(test_set, selected_idx)
            loader = DataLoader(subset, batch_size=train_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
            with torch.no_grad():
                for images, lbl in loader:
                    images = images.to(device, non_blocking=True)
                    _ = model(images)
                    feat_batch = outputs["feat"]
                    feat_batch = torch.flatten(feat_batch, 1).cpu()
                    feats.append(feat_batch)
                    labels.append(lbl)
                    task_ids.append(torch.full_like(lbl, task_id))
    finally:
        handle.remove()

    feat_arr = torch.cat(feats, dim=0).numpy()
    label_arr = torch.cat(labels, dim=0).numpy()
    task_arr = torch.cat(task_ids, dim=0).numpy()
    np.savez(feature_save_path, feat=feat_arr, label=label_arr, task_id=task_arr,
             backbone=np.array([backbone_name] * len(label_arr)))
    print(f"[Feature] saved {feature_save_path}, feat_shape={feat_arr.shape}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbones", type=str, nargs="+", default=["proxyless_subnet", "resnet18", "small_model"])
    p.add_argument("--num_tasks", type=int, default=5)
    p.add_argument("--classes_per_task", type=int, default=20)
    p.add_argument("--proxyless_config", type=str, default=None, help="Proxyless 子网 config 路径，未提供则读取环境变量 PROXYLESS_SUBNET_CONFIG")
    p.add_argument("--dataset_location", type=str, required=True)
    p.add_argument("--dataset", type=str, default="CIFAR100", choices=["CIFAR100", "CIFAR10"])
    p.add_argument("--train_batch_size", type=int, default=128)
    p.add_argument("--test_batch_size", type=int, default=256)
    p.add_argument("--n_worker", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=2)
    p.add_argument("--init_lr", type=float, default=0.02)
    p.add_argument("--replay_capacity", type=int, default=5000)
    p.add_argument("--replay_per_batch", type=int, default=64)
    p.add_argument("--replay_mode", type=str, default="age_priority")
    p.add_argument("--replay_old_task_scale", type=float, default=1.0)
    p.add_argument("--replay_old_task_scale_by_F", type=float, default=0.0)
    p.add_argument("--replay_priority_mode", type=str, default="forgetting")
    # KD / EWC 超参（与主实验保持一致）
    p.add_argument("--cl_kd_method", type=str, default="logit")
    p.add_argument("--cl_kd_logit_lambda", type=float, default=1.0)
    p.add_argument("--cl_kd_temperature", type=float, default=2.0)
    p.add_argument("--cl_kd_conf_threshold", type=float, default=0.5)
    p.add_argument("--cl_ortho_method", type=str, default="none")
    p.add_argument("--cl_ortho_scale", type=float, default=1.0)
    p.add_argument("--ortho_samples_per_task", type=int, default=2048)
    p.add_argument("--cl_reg_method", type=str, default="ewc")
    p.add_argument("--cl_reg_decay", type=float, default=1.0)
    p.add_argument("--cl_reg_clip", type=float, default=None)
    p.add_argument("--cl_penalty_clip", type=float, default=None)
    p.add_argument("--ewc_lambda", type=float, default=0.0)
    p.add_argument("--ewc_samples_per_task", type=int, default=0)
    p.add_argument("--ewc_online_interval", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="./cil_tsne_outputs")
    p.add_argument("--samples_per_class", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if args.dataset.lower() == "cifar10" else 100
    if args.num_tasks * args.classes_per_task != num_classes:
        raise ValueError(f"{args.dataset} 需要 num_tasks × classes_per_task = {num_classes}，请检查参数设置")
    tasks = build_task_indices(num_classes=num_classes, num_tasks=args.num_tasks, classes_per_task=args.classes_per_task, seed=0)
    feature_paths = []
    for backbone in args.backbones:
        ckpt = train_single_client(backbone, args, device, args.save_dir, tasks)
        feat_path = os.path.join(args.save_dir, f"features_{backbone}.npz")
        extract_features_for_backbone(
            backbone, ckpt, feat_path, dataset_root=args.dataset_location,
            tasks=tasks, samples_per_class=args.samples_per_class,
            train_batch_size=args.test_batch_size, num_workers=args.n_worker,
            proxyless_config=args.proxyless_config, dataset=args.dataset,
        )
        feature_paths.append(feat_path)
    print("Done. Feature files:", feature_paths)


if __name__ == "__main__":
    main()
