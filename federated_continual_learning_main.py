#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import copy
import glob
import os
import pickle  # 仍保留，可能下游用到
import time
import warnings
import inspect
import json
import sys

import pynvml
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")  # 与原逻辑一致

from auto_resume import AutoResumeManager, atomic_torch_save
from arch_prior import (
    ClientHistory,
    append_arch_prior_log,
    build_arch_prior_weights,
    build_fused_arch_parameters,
    clone_arch_parameters,
    compute_similarity_scores,
    compute_task_prototype,
    export_supernet_client_subnet,
    load_client_histories,
    load_subnet_from_artifact,
    save_client_histories,
    summarize_arch_parameters,
)
from clustering_machine import *
from data_providers.cifar100_fcl_dirichlet_split import CifarDataProvider100
from nas_manager import ArchSearchRunManager, GradientArchSearchConfig, RLArchSearchConfig
from retrain_pipeline import (
    RetrainPipelineHelpers,
    run_baseline_retrain_pipeline,
    run_supernet_retrain_pipeline,
)
from runtime_context import RuntimeContext
from utils_old import *
from models.super_nets.super_proxyless import *
from models.normal_nets.proxyless_nets import *
from models.baseline_nets import BaselineResNet
from utils.pytorch_utils import create_exp_dir
from utils.pytorch_utils import accuracy
from run_manager import CifarRunConfig, SimpleReplayBuffer

warnings.filterwarnings("ignore")

from commonwealth_machine import *

# import torch
# torch.cuda.set_per_process_memory_fraction(0.8, 0)


# ----------------------------- 常量与工具函数 -----------------------------

# 参考值常量：保持与原始值一致
REF_VALUES = {
    "flops": {
        "0.35": 59 * 1e6,
        "0.50": 97 * 1e6,
        "0.75": 209 * 1e6,
        "1.00": 300 * 1e6,
        "1.30": 509 * 1e6,
        "1.40": 582 * 1e6,
    },
    # ms
    "mobile": {"1.00": 80},
    "cpu": {"1.00": 6},
    "gpu8": {"1.00": 65},
}

def line_info() -> str:
    """返回'[文件:行号]'，用于统一打印位置。"""
    f = inspect.currentframe().f_back
    return f"[{f.f_code.co_filename}:{f.f_lineno}]"

def model_signature(model: torch.nn.Module) -> dict:
    """返回可打印的模型关键统计，便于判断是否继承成功或被重置."""
    state = model.state_dict()
    # 选取首层卷积和分类头，若不存在则返回空
    sig = {}
    for k in ["first_conv.conv.weight", "classifier.linear.weight"]:
        if k in state:
            t = state[k].float()
            sig[k] = {
                "shape": list(t.shape),
                "mean": float(t.mean()),
                "std": float(t.std()),
                "norm": float(t.norm()),
            }
    # 全模型范数作为粗粒度检查
    flat = torch.cat([p.flatten().float() for p in state.values() if p.dtype.is_floating_point])
    sig["global"] = {"num_params": flat.numel(), "norm": float(flat.norm()), "mean": float(flat.mean()), "std": float(flat.std())}
    return sig

def set_target_hardware(idx: int):
    """原项目已有同名函数时会覆盖；若无，则此占位用于类型提示。"""
    return ["mobile", "cpu", "gpu8", "flops", None][idx % 5]


def parse_optional_bool(value):
    if value is None or isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_prev_task(super_net, prev_task_path: str):
    """
    加载上一任务保存的超网权重，用于连续任务的初始化。
    查找顺序：checkpoint/latest.txt -> global.pth.tar -> warmup.pth.tar。
    """
    ckpt_dir = os.path.join(prev_task_path, "checkpoint")
    candidates = []
    latest_txt = os.path.join(ckpt_dir, "latest.txt")            # 优先级1：checkpoint（断点）保存的超网
    if os.path.isfile(latest_txt):
        try:
            with open(latest_txt, "r") as fin:
                candidates.append(fin.readline().strip())
        except Exception:
            pass
        
    candidates.append(os.path.join(ckpt_dir, "global.pth.tar"))     # 优先级2：search阶段的保存的超网
    candidates.append(os.path.join(ckpt_dir, "warmup.pth.tar"))     # 优先级3：warmup阶段的保存的超网

    for path in candidates:
        if path and os.path.isfile(path):
            try:
                checkpoint = torch.load(path, map_location=torch.device("cpu"))
                model_dict = super_net.state_dict()
                model_dict.update(checkpoint.get("state_dict", {}))
                super_net.load_state_dict(model_dict)
                print(f"[FCL] Loaded previous task supernet from {path}")
                return True
            except Exception as e:
                print(f"[FCL] Failed to load supernet from {path}")
                continue
    print(f"[FCL] No previous task checkpoint found under {prev_task_path}, start from scratch.")
    return False


def load_task1_bootstrap(super_net, bootstrap_ckpt_path: str):
    """
    仅用于 task1：从外部超网 checkpoint 初始化 supernet。
    支持 `{"state_dict": ...}` 或直接 state_dict 结构。
    """
    if not bootstrap_ckpt_path:
        return False
    ckpt_path = os.path.abspath(bootstrap_ckpt_path)
    if not os.path.isfile(ckpt_path):
        print(f"[FCL] Task1 bootstrap checkpoint not found: {ckpt_path}, start task1 from scratch.")
        return False
    try:
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = super_net.state_dict()
        model_dict.update(state_dict)
        super_net.load_state_dict(model_dict)
        print(f"[FCL] Loaded task1 bootstrap supernet from {ckpt_path}")
        return True
    except Exception as e:
        print(f"[FCL] Failed to load task1 bootstrap supernet from {ckpt_path}: {e}")
        return False


def load_prev_task_optimizers(run_mgr, prev_task_path: str, client_id: int, task_id_from: int, is_server: bool = False):
    """
    尝试从上一任务的 checkpoint 中恢复当前 run_mgr 的优化器状态。
    仅在 checkpoint 存在且键匹配时加载；否则静默跳过。
    """
    ckpt_path = os.path.join(prev_task_path, "checkpoint", "global.pth.tar")
    if not os.path.isfile(ckpt_path):
        return False
    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"[FCL] Failed to load prev optimizer ckpt {ckpt_path}: {e}")
        return False

    if is_server:
        w_key = "server_weight_optimizer"
        a_key = "server_arch_optimizer"
    else:
        w_key = f"task_{task_id_from}_{client_id}_weight_optimizer"
        a_key = f"task_{task_id_from}_{client_id}_arch_optimizer"
    loaded = False
    if hasattr(run_mgr, "run_manager") and hasattr(run_mgr.run_manager, "optimizer"):
        if w_key in ckpt:
            try:
                run_mgr.run_manager.optimizer.load_state_dict(ckpt[w_key])
                loaded = True
                print(f"[FCL] Loaded weight optimizer for client {client_id} from {ckpt_path}")
            except Exception as e:
                print(f"[FCL] Failed to load weight optimizer for client {client_id}: {e}")
    if hasattr(run_mgr, "arch_optimizer"):
        if a_key in ckpt:
            try:
                run_mgr.arch_optimizer.load_state_dict(ckpt[a_key])
                loaded = True
                print(f"[FCL] Loaded arch optimizer for client {client_id} from {ckpt_path}")
            except Exception as e:
                print(f"[FCL] Failed to load arch optimizer for client {client_id}: {e}")
    return loaded


def _load_state_with_fallback(primary_path: str, fallback_path: str, desc: str):
    """按主路径→备份路径顺序加载 state dict，保持打印可见，出错不抛出。"""
    for p, tag in [(primary_path, "current"), (fallback_path, "prev")]:
        if not p:
            continue
        if os.path.isfile(p):
            try:
                state = torch.load(p, map_location="cpu")
                print(f"[{desc}] Loaded {tag} state from {p}")
                return state
            except Exception as e:
                print(f"[{desc}] Failed to load {tag} state from {p}: {e}")
    return None


def _save_state_safely(state, path: str, desc: str):
    """保存 state dict，保持打印，失败不抛出。"""
    if state is None:
        return 0.0
    start = time.time()
    try:
        atomic_torch_save(state, path)
        print(f"[{desc}] Saved state to {path}")
    except Exception as e:
        print(f"[{desc}] Failed to save state to {path}: {e}")
    return max(0.0, time.time() - start)


def _cleanup_search_checkpoint_files(task_path: str, *, remove_global: bool, remove_warmup: bool):
    ckpt_dir = os.path.join(task_path, "checkpoint")
    if not os.path.isdir(ckpt_dir):
        return []
    removed = []
    targets = []
    if remove_global:
        targets.append("global.pth.tar")
    if remove_warmup:
        targets.append("warmup.pth.tar")
    for name in targets:
        path = os.path.join(ckpt_dir, name)
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(path)
            except Exception as e:
                print(f"[CheckpointCleanup] Failed to remove {path}: {e}")
    latest_txt = os.path.join(ckpt_dir, "latest.txt")
    if os.path.isfile(latest_txt):
        try:
            with open(latest_txt, "r") as fin:
                latest_target = fin.readline().strip()
        except Exception:
            latest_target = ""
        if latest_target and any(os.path.abspath(latest_target) == os.path.abspath(p) for p in removed):
            try:
                os.remove(latest_txt)
            except Exception as e:
                print(f"[CheckpointCleanup] Failed to remove stale latest.txt {latest_txt}: {e}")
    if removed:
        print(f"[CheckpointCleanup] Removed {len(removed)} obsolete checkpoint files under {ckpt_dir}")
    return removed


def load_previous_subnet_for_evaluation(learned_net_path: str):
    net_config_path = os.path.join(learned_net_path, "net.config")
    if not os.path.isfile(net_config_path):
        return None
    try:
        from models import get_net_by_name

        net_config = json.load(open(net_config_path, "r"))
        subnet = get_net_by_name(net_config["name"]).build_from_config(net_config)
        if not load_prev_task(subnet, learned_net_path):
            return None
        return subnet
    except Exception as e:
        print(f"[QuickEval] Failed to prepare previous subnet from {learned_net_path}: {e}")
        return None


def quick_evaluate_previous_subnet(previous_subnet, run_config, max_batches: int = 1):
    if previous_subnet is None:
        return None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(previous_subnet).to(device)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    losses = AverageMeter()
    top1 = AverageMeter()
    processed_batches = 0
    with torch.no_grad():
        for images, labels in run_config.train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, labels)
            acc1, _ = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            processed_batches += 1
            if processed_batches >= max_batches:
                break
    if processed_batches == 0:
        return None
    return {
        "loss": losses.avg,
        "top1": top1.avg,
        "batches": processed_batches,
        "client_id": getattr(run_config, "client_id", None),
    }


def run_nas_search_for_task(
    args,
    task_id,
    global_server,
    clients,
    client_indices,
    prev_task_path=None,
    current_task_path=None,
    client_histories=None,
):
    # TODO: 使用历史任务原型初始化架构参数 alpha。
    # TODO: 在此处接入成本约束与搜索门控。
    if (
        getattr(args, "enable_arch_prior", False)
        and not getattr(args, "resume", False)
        and client_histories is not None
        and prev_task_path is not None
        and current_task_path is not None
    ):
        arch_prior_log_path = os.path.join(current_task_path, "logs", "arch_prior_details.jsonl")
        client_artifact_dir = os.path.join(current_task_path, "client_subnets")
        os.makedirs(client_artifact_dir, exist_ok=True)
        fallback_learned_net_path = os.path.join(prev_task_path, "learned_net", "net.config")
        for idx in client_indices:
            history = client_histories[idx]
            client = clients[idx]
            client.arch_prior_state = None
            client.arch_prior_regularization_target = None
            client.arch_prior_regularization_enabled = False
            client.arch_prior_regularization_lambda = 0.0

            previous_subnet = None
            prev_client_artifact = history.subnet_artifacts.get(task_id - 1) if history is not None else None
            if prev_client_artifact is not None:
                previous_subnet = load_subnet_from_artifact(prev_client_artifact)
            if previous_subnet is None and os.path.isfile(fallback_learned_net_path):
                previous_subnet = load_previous_subnet_for_evaluation(os.path.join(prev_task_path, "learned_net"))

            current_proto = None
            if previous_subnet is not None:
                try:
                    current_proto = compute_task_prototype(
                        previous_subnet,
                        client.run_manager.run_config.train_loader,
                        client.run_manager.device,
                    )
                except Exception as e:
                    print(f"[ArchPrior] Failed to compute prototype for client {idx} task {task_id}: {e}")

            similarity_scores = {}
            selected_task_ids = []
            weights = None
            fused_arch_params = None
            if current_proto is not None and history.has_any():
                similarity_scores = compute_similarity_scores(current_proto, history.task_prototypes)
                selected_task_ids, weights = build_arch_prior_weights(
                    current_proto,
                    history.task_prototypes,
                    topk=getattr(args, "arch_prior_topk", 3),
                    tau=getattr(args, "arch_prior_tau", 1.0),
                )
                fused_arch_params = build_fused_arch_parameters(
                    history.arch_parameters,
                    selected_task_ids,
                    weights,
                )
                if fused_arch_params:
                    client.arch_prior_state = {
                        "task_id": int(task_id),
                        "client_id": int(idx),
                        "current_proto": current_proto.detach().cpu().clone(),
                        "selected_task_ids": [int(task_i) for task_i in selected_task_ids],
                        "weights": weights.detach().cpu().clone() if weights is not None else None,
                        "similarity_scores": similarity_scores,
                        "fused_arch_params": fused_arch_params,
                        "artifact_path": os.path.join(client_artifact_dir, f"client_{idx}_task_{task_id}_subnet.pt"),
                        "applied": False,
                        "alpha_before_stats": summarize_arch_parameters(clone_arch_parameters(client.net)),
                    }

            if getattr(args, "log_arch_prior_details", True):
                proto_stats = {}
                if current_proto is not None:
                    proto_float = current_proto.detach().float().cpu()
                    proto_stats = {
                        "current_proto_norm": float(proto_float.norm().item()),
                        "current_proto_mean": float(proto_float.mean().item()),
                        "current_proto_var": float(proto_float.var(unbiased=False).item()),
                    }
                append_arch_prior_log(
                    arch_prior_log_path,
                    {
                        "event": "prior_prepared",
                        "task_id": int(task_id),
                        "client_id": int(idx),
                        "enabled": True,
                        "has_history": bool(history.has_any()),
                        "has_previous_subnet": previous_subnet is not None,
                        "selected_task_ids": [int(task_i) for task_i in selected_task_ids],
                        "weights": weights.detach().cpu().tolist() if weights is not None else None,
                        "similarity_scores": {int(k): float(v) for k, v in similarity_scores.items()},
                        "topk": int(getattr(args, "arch_prior_topk", 3)),
                        "tau": float(getattr(args, "arch_prior_tau", 1.0)),
                        "enable_arch_prior_loss": bool(getattr(args, "enable_arch_prior_loss", False)),
                        "arch_prior_loss_lambda": float(getattr(args, "arch_prior_loss_lambda", 0.0)),
                        "alpha_before_stats": summarize_arch_parameters(clone_arch_parameters(client.net)),
                        **proto_stats,
                    },
                )

    search_machine = ClusteringMachine(
        target_hardware="super_net",
        config=args,
        global_server=global_server,
        clients_idx_arr=client_indices,
        clients=clients,
        start_round=args.start_round,
        last_round=args.last_round,
        path=args.path,
        task_id=task_id,
    )
    search_machine.run()
    if current_task_path is not None:
        client_artifact_dir = os.path.join(current_task_path, "client_subnets")
        os.makedirs(client_artifact_dir, exist_ok=True)
        should_update_prior_history = (
            getattr(args, "enable_arch_prior", False) and client_histories is not None
        )
        arch_prior_log_path = os.path.join(current_task_path, "logs", "arch_prior_details.jsonl")

        for idx in client_indices:
            client = clients[idx]
            artifact_path = os.path.join(client_artifact_dir, f"client_{idx}_task_{task_id}_subnet.pt")
            artifact_export_dir = os.path.join(client_artifact_dir, f"client_{idx}_task_{task_id}")
            normal_net = export_supernet_client_subnet(
                client,
                artifact_path,
                {
                    "width_stages": args.width_stages,
                    "n_cell_stages": args.n_cell_stages,
                    "stride_stages": args.stride_stages,
                    "conv_candidates": args.conv_candidates,
                    "n_classes": client.run_manager.run_config.data_provider.n_classes,
                    "width_mult": args.width_mult,
                    "bn_param": (args.bn_momentum, args.bn_eps),
                    "dropout_rate": args.dropout,
                    "inference_device": args.object_to_search,
                },
                export_dir=artifact_export_dir,
            )

            if not should_update_prior_history:
                continue

            history = client_histories[idx]
            prior_state = getattr(client, "arch_prior_state", None) or {}
            current_proto = prior_state.get("current_proto")
            if current_proto is None:
                try:
                    current_proto = compute_task_prototype(
                        normal_net,
                        client.run_manager.run_config.train_loader,
                        client.run_manager.device,
                    )
                except Exception as e:
                    print(f"[ArchPrior] Failed to compute post-search prototype for client {idx} task {task_id}: {e}")
                    current_proto = None
            arch_params = clone_arch_parameters(client.net)
            if current_proto is not None:
                history.task_prototypes[int(task_id)] = current_proto.detach().cpu().clone()
            history.arch_parameters[int(task_id)] = arch_params
            history.subnet_artifacts[int(task_id)] = artifact_path
            if getattr(args, "log_arch_prior_details", True):
                proto_stats = {}
                if current_proto is not None:
                    proto_float = current_proto.detach().float().cpu()
                    proto_stats = {
                        "stored_proto_norm": float(proto_float.norm().item()),
                        "stored_proto_mean": float(proto_float.mean().item()),
                        "stored_proto_var": float(proto_float.var(unbiased=False).item()),
                    }
                append_arch_prior_log(
                    arch_prior_log_path,
                    {
                        "event": "history_updated",
                        "task_id": int(task_id),
                        "client_id": int(idx),
                        "artifact_path": artifact_path,
                        "history_size": len(history.arch_parameters),
                        "stored_arch_stats": summarize_arch_parameters(arch_params),
                        **proto_stats,
                    },
                )

        if should_update_prior_history:
            save_client_histories(
                client_histories,
                os.path.join(current_task_path, "client_arch_prior_histories.pt"),
            )
    return search_machine.get_server()


def train_personalized_subnet(args, global_run_manager, clients, client_indices, start_round, last_round):
    # TODO: 在此处接入个性化元学习初始化。
    # NAS 子网重训阶段固定为“仅本地训练，不做聚合”。
    args.retrain_fedavg = False
    retrain_machine = CommonwealthMachine(
        target_hardware="supernet",
        config=args,
        global_run_manager=global_run_manager,
        clients_idx_arr=client_indices,
        clients=clients,
        start_round=start_round,
        last_round=last_round,
        path=args.path,
    )
    retrain_machine.run()
    return retrain_machine.get_server()

# ----------------------------- 参数解析与派生 -----------------------------
def parse_args() -> argparse.Namespace:
    print("set格式化参数开始...")
    parser = argparse.ArgumentParser()

    # Federated Learning
    parser.add_argument("--gpu", default="0,1,2,3", help="gpu available to use")
    parser.add_argument("--num_users", type=int, default=10, help="number of clients: K")
    parser.add_argument("--num_tasks", type=int, default=10, help="number of tasks: K")
    parser.add_argument("--start_task_id", type=int, default=1, help="start task id")
    parser.add_argument("--end_task_id", type=int, default=None, help="end task id for this run; default uses num_tasks")
    parser.add_argument("--object_to_search", type=str, default="supernet",
                        choices=["supernet", "cpu", "gpu8", "flops", "baseline"],
                        help="search target(0:supernet, 1:cpu, 2:gpu8, 3:flops) latency for 1/2 , compute source for 3.")
    parser.add_argument("--iid", type=int, default=0,
                        help="Default set 1 to IID. Set to 0 for non-IID.")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3,
                        help="Dirichlet alpha for non-IID client split. Ignored when --iid 1.")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation split ratio inside each client/task partition.")
    parser.add_argument("--unequal", type=int, default=1,
                        help="whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)")

    # ProxylessNAS
    parser.add_argument("--warmup", action="store_true", help="if have not warmup(only pretrain supernet, don't nas), please set it True")
    parser.add_argument("--path", type=str, default="./output/proxyless-", help="checkpoint save path")
    parser.add_argument("--task1_bootstrap_ckpt", type=str, default="./global.pth.tar",
                        help="task1 启动时的超网初始化 checkpoint 路径；文件存在时自动加载，仅作用于 task1")
    parser.add_argument("--save_env", type=str, default="EXP", help="experiment time name to save exp code")
    parser.add_argument("--resume", action="store_true", help="load last checkpoint")
    parser.add_argument("-R", "--auto_resume", action="store_true",
                        help="自动从最近一次安全保存点继续训练，自动判断 task/phase/round")
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed to make experiments reproducible")
    parser.add_argument("--start_round", default=0, type=int, help="start round in fed_search")
    parser.add_argument("--last_round", default=10, type=int, help="last round in fed_search. 125 for all clients. 175 for cpu/gpu.")  # 讨论硬件差异时就多训练些联邦轮次
    parser.add_argument("--retrain_start_round", type=int, default=0, help="重训阶段起始轮次，默认沿用 start_round")
    parser.add_argument("--retrain_last_round", type=int, default=20, help="重训阶段最后轮次，默认沿用 last_round")
    parser.add_argument("--retrain_sequence_from_task1", action="store_true",
                        help="重训阶段依次从 task1 训练到当前任务，每个任务各跑 retrain_last_round 轮")
    parser.add_argument("--retrain_fedavg", type=parse_optional_bool, nargs="?", const=True, default=False,
                        help="重训阶段是否启用 FedAvg 聚合；默认 False（用于 NAS 子网个性化本地重训）")
    parser.add_argument("--federate_arch_params", type=parse_optional_bool, nargs="?", const=True, default=False,
                        help="是否在联邦阶段聚合并下发架构参数 alpha(AP_path_alpha/AP_path_wb)；默认 False（客户端独立持有架构参数）")
    parser.add_argument("--enable_arch_prior", action="store_true",
                        help="是否在 NAS 搜索前启用历史先验引导的架构参数初始化，默认关闭")
    parser.add_argument("--arch_prior_topk", type=int, default=3,
                        help="历史先验 Top-K 任务数")
    parser.add_argument("--arch_prior_tau", type=float, default=0.5,
                        help="历史先验 softmax 温度参数")
    parser.add_argument("--enable_arch_prior_loss", action="store_true",
                        help="是否在搜索阶段的架构参数更新中加入与历史先验偏移的正则项，默认关闭")
    parser.add_argument("--arch_prior_loss_lambda", type=float, default=0.0,
                        help="历史先验偏移正则强度，仅对 gradient-based NAS 生效")
    parser.add_argument("--log_arch_prior_details", type=parse_optional_bool, default=True,
                        help="是否记录历史先验的详细中间变量和统计量")
    
    parser.add_argument("--cl_kd_method", type=str, default="none",
                        choices=["none", "logit", "logit_conf"],
                        help="持续学习蒸馏方法：none / logit / logit_conf")
    parser.add_argument("--cl_kd_logit_lambda", type=float, default=0.0,
                        help="logit 蒸馏 loss 权重，0 表示关闭蒸馏")
    parser.add_argument("--cl_kd_temperature", type=float, default=2.0,
                        help="logit 蒸馏的温度系数 T")
    parser.add_argument("--cl_kd_conf_threshold", type=float, default=0.5,
                        help="logit_conf 模式下 teacher 置信度阈值，低于该值的类不参与 KD")
    parser.add_argument("--replay_mode", type=str, default="none", choices=["none", "global", "task_balanced", "age_priority"], help="experience replay 模式：none/global/task_balanced/age_priority")
    parser.add_argument("--replay_capacity", type=int, default=0, help="重放缓冲区最大样本量（全局计数）")
    parser.add_argument("--replay_capacity_ratio", type=float, default=None, help="按全训练集样本数的比例设置缓冲区容量（0~1），高于 replay_capacity 时覆盖之")
    parser.add_argument("--replay_per_batch", type=int, default=0, help="每个 batch 从缓冲区重放的样本数")
    parser.add_argument("--replay_old_task_scale", type=float, default=1.0, help="age_priority 模式下旧任务样本的权重缩放，>1 让旧任务更容易被采样")
    parser.add_argument("--replay_old_task_scale_by_F", type=float, default=0.0, help="按遗忘程度动态放大旧任务样本权重，0 表示不启用，单位：每个遗忘点数的放大系数")
    parser.add_argument("--enable_replay", action="store_true",
                        help="是否在子网重训阶段启用 replay 模块，默认关闭")
    parser.add_argument("--enable_kd", action="store_true",
                        help="是否在子网重训阶段启用 logit KD 模块，默认关闭")
    
    
    # 分阶段可选覆盖：当前仅保留 retrain_*；search 阶段的遗忘相关覆盖已停用
    parser.add_argument("--retrain_cl_kd_method", type=str, default=None, choices=["none","logit","logit_conf"])
    parser.add_argument("--retrain_cl_kd_logit_lambda", type=float, default=None)
    parser.add_argument("--retrain_cl_kd_temperature", type=float, default=None)
    parser.add_argument("--retrain_cl_kd_conf_threshold", type=float, default=None)
    parser.add_argument("--retrain_enable_kd", type=parse_optional_bool, default=None)
    parser.add_argument("--retrain_replay_mode", type=str, default=None, choices=["none", "global", "task_balanced", "age_priority"])
    parser.add_argument("--retrain_replay_capacity", type=int, default=None)
    parser.add_argument("--retrain_replay_capacity_ratio", type=float, default=None)
    parser.add_argument("--retrain_replay_per_batch", type=int, default=None)
    parser.add_argument("--retrain_replay_old_task_scale", type=float, default=None)
    parser.add_argument("--retrain_replay_old_task_scale_by_F", type=float, default=None)
    

    parser.add_argument("--local_epoch_number", default=5, type=int, help="local epoch each round in fed_search,during each epoch all data will be trained once")

    # run config
    parser.add_argument("--client_id", type=int, default=10, help="local single client id")
    parser.add_argument("--dataset_location", type=str, default="/dataset/cifar10/", help="cifar dataset path. e.g. /dataset/cifar10/ ")
    parser.add_argument("--n_epochs", type=int, default=500,
                        help="local clients full epoch numbers on single client,for single client training "
                             "equal to local_epoch_number * (last_round - start_round)")
    parser.add_argument("--init_lr", type=float, default=0.006, help="init learning rate for parameter update, if too large may lead to unstable or diverge training, if too small may lead to slow converge or stuck in local optimal. The learning rate should be tuned together with optimizer(momentum/weight_decay), batch_size and lr_schedule(cosine/warmup). ")
    parser.add_argument("--lr_schedule_type", type=str, default="cosine", help ="learning rate decay policy")

    parser.add_argument("--dataset", type=str, default="CIFAR100", choices=["CIFAR10", "CIFAR100"], help="dataset type")
    parser.add_argument("--train_batch_size", type=int, default=1024, help="training batch size for each client, the number of picture load in memory during one training iteration")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="testing batch size for each client, the number of picture load in memory during one testing iteration")
    parser.add_argument("--valid_size", type=int, default=50000, help="validation size for each client during training for imagenet")

    parser.add_argument("--opt_type", type=str, default="sgd", choices=["sgd"], help="optimizer type for update weights and bias to minimize loss")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum factor for sgd optimizer")
    parser.add_argument("--no_nesterov", action="store_true", help="do not use nesterov momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay (L2 penalty) for optimizer")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="label smoothing value for loss function")
    parser.add_argument("--no_decay_keys", type=str, default=None, choices=[None, "bn", "bn#bias"], help="no weight decay on batch norm and bias")

    parser.add_argument("--model_init", type=str, default="he_fout", choices=["he_fin", "he_fout"], help ="convolution weight initialization method")
    parser.add_argument("--init_div_groups", action="store_true", help="whether to divide the initialization range by number of groups in conv layer")
    parser.add_argument("--validation_frequency", type=int, default=2, help="validate model per n epochs during training")
    parser.add_argument("--print_frequency", type=int, default=10,help="print training log per n iterations during training")
    parser.add_argument("--n_worker", type=int, default=2,help="number of workers during data loading")  # 1 is most stable. 2 or 4 is bad. 3 is also ok.
    parser.add_argument("--search", action="store_true", help="use it in search")
    parser.add_argument("--baseline_arch", type=str, default="resnet18",
                        help="fixed backbone name in torchvision.models, e.g. resnet18/resnet34/resnet50/mobilenet_v2")
    parser.add_argument("--baseline_pretrained", action="store_true",
                        help="use torchvision pretrained weights for baseline backbones")
    parser.add_argument("--baseline_method", type=str, default="fedavg",
                        choices=["fedavg", "target", "re_fed", "ditto", "fedweit", "re-fed"],
                        help="baseline 策略：fedavg / target / re_fed / ditto / fedweit")
    parser.add_argument("--baseline_auto_config", type=parse_optional_bool, nargs="?", const=True, default=True,
                        help="是否对 baseline_method 自动填充推荐超参（replay/KD 等），默认 True")
    parser.add_argument("--baseline_replay_mode", type=str, default="task_balanced",
                        choices=["none", "global", "task_balanced", "age_priority"],
                        help="baseline 自动配置 replay 时使用的模式")
    parser.add_argument("--baseline_replay_capacity_ratio", type=float, default=0.1,
                        help="baseline 自动配置 replay 时的容量比例（仅在未显式设置 replay_capacity/replay_capacity_ratio 时生效）")
    parser.add_argument("--baseline_replay_per_batch", type=int, default=32,
                        help="baseline 自动配置 replay 时每 batch 重放样本数（仅在 replay_per_batch<=0 时生效）")
    parser.add_argument("--baseline_target_kd_method", type=str, default="logit",
                        choices=["none", "logit", "logit_conf"],
                        help="baseline_method=target 自动配置 KD 时的蒸馏方式")
    parser.add_argument("--baseline_target_kd_lambda", type=float, default=1.0,
                        help="baseline_method=target 自动配置 KD 权重（仅在 cl_kd_logit_lambda<=0 时生效）")
    parser.add_argument("--baseline_target_kd_temperature", type=float, default=2.0,
                        help="baseline_method=target 自动配置 KD 温度（仅在 cl_kd_temperature<=0 时生效）")
    parser.add_argument("--ditto_mu", type=float, default=0.01,
                        help="baseline_method=ditto 时的 proximal 正则系数")
    parser.add_argument("--fedweit_personal_keys", type=str,
                        default="backbone.fc.weight,backbone.fc.bias,fc.weight,fc.bias,classifier.weight,classifier.bias,classifier.1.weight,classifier.1.bias,backbone.classifier.1.weight,backbone.classifier.1.bias,linear.weight,linear.bias",
                        help="baseline_method=fedweit 时不参与全局聚合、保持客户端个性化的参数键（逗号分隔）")

    # supernet config（看代码细节了解搜索空间的工作原理）
    parser.add_argument("--width_stages", type=str, default="24,40,80,96,192,320",
                        help="width (output channels) of each cell case in the block, "
                             "also last_channel = make_divisible(400 * width_mult, 8) if width_mult > 1.0 else 400")
    parser.add_argument("--n_cell_stages", type=str, default="2,3,4,3,4,3",
                        help="number of cells in each cell case")
    parser.add_argument("--stride_stages", type=str, default="1,1,2,1,2,1",
                        help="stride of each cell case in the block")
    parser.add_argument("--width_mult", type=float, default=1.0,help="width multiplier for model")
    parser.add_argument("--bn_momentum", type=float, default=0.1,help="batch norm momentum")
    parser.add_argument("--bn_eps", type=float, default=1e-3,help="batch norm epsilon")
    parser.add_argument("--dropout", type=float, default=0,help="dropout rate")

    # architecture search config
    parser.add_argument("--arch_algo", type=str, default="grad", choices=["grad", "rl"],help="architecture search algorithm")
    parser.add_argument("--warmup_n_rounds", type=int, default=5, help="warmup rounds to pretrain supernet before architecture search")
    parser.add_argument("--skip_warmup", action="store_true",
                        help="跳过 warmup 阶段，直接进入 search，用于快速验证")
    parser.add_argument("--skip_search", action="store_true",
                        help="跳过 warmup/search，直接进入 learned_net 重训（要求已有 learned_net/init 和 net.config）")
    parser.add_argument("--retrain_only", action="store_true",
                        help="仅执行已固化子网的重训流程，不进入 warmup/search")
    parser.add_argument("--auto_cleanup_warmup_ckpt", dest="auto_cleanup_warmup_ckpt", action="store_true", default=True,
                        help="在当前任务 search 成功结束后自动删除 checkpoint/warmup.pth.tar，默认开启")
    parser.add_argument("--no_auto_cleanup_warmup_ckpt", dest="auto_cleanup_warmup_ckpt", action="store_false",
                        help="关闭当前任务 search 完成后的 warmup checkpoint 自动清理")
    parser.add_argument("--auto_cleanup_prev_task_search_ckpt", dest="auto_cleanup_prev_task_search_ckpt", action="store_true", default=True,
                        help="在当前任务 search 成功结束后自动删除上一任务 checkpoint/global.pth.tar 和 warmup.pth.tar，默认开启")
    parser.add_argument("--no_auto_cleanup_prev_task_search_ckpt", dest="auto_cleanup_prev_task_search_ckpt", action="store_false",
                        help="关闭上一任务 search checkpoint 自动清理")

    # shared hyper-parameters
    parser.add_argument("--arch_init_type", type=str, default="normal", choices=["normal", "uniform"],help="initialization type for architecture parameters")
    parser.add_argument("--arch_init_ratio", type=float, default=1e-3,help="initialization ratio for architecture parameters")
    parser.add_argument("--arch_opt_type", type=str, default="adam", choices=["adam"],help="optimizer type for architecture parameters")
    parser.add_argument("--arch_lr", type=float, default=1e-3,help="learning rate for architecture parameters")
    parser.add_argument("--arch_adam_beta1", type=float, default=0,help="beta1 for adam optimizer")
    parser.add_argument("--arch_adam_beta2", type=float, default=0.999,help="beta2 for adam optimizer")
    parser.add_argument("--arch_adam_eps", type=float, default=1e-8,help="epsilon for adam optimizer")
    parser.add_argument("--arch_weight_decay", type=float, default=0,help="weight decay for architecture parameters")
    parser.add_argument("--target_hardware", type=str, default=None,
                        choices=["mobile", "cpu", "gpu8", None, "flops"],help="target hardware for architecture search")

    # Grad hyper-parameters
    parser.add_argument("--grad_update_arch_param_every", type=int, default=5,help="update architecture parameters every N steps")
    parser.add_argument("--grad_update_steps", type=int, default=1,help="number of update steps for architecture parameters")
    parser.add_argument("--grad_binary_mode", type=str, default="two",
                        choices=["full_v2", "full", "two"], help="binary mode to sample paths for gradient estimation")
    parser.add_argument("--grad_data_batch", type=int, default=None, help="batch size for architecture parameters update, if None use train_batch_size")
    parser.add_argument("--grad_reg_loss_type", type=str, default="add#linear",
                        choices=["add#linear", "mul#log"], help="regularization loss type")
    parser.add_argument("--grad_reg_loss_lambda", type=float, default=0.05, help="regularization loss lambda")
    parser.add_argument("--grad_reg_loss_alpha", type=float, default=0.2, help="regularization loss alpha")
    parser.add_argument("--grad_reg_loss_beta", type=float, default=0.3, help="regularization loss beta")

    # RL hyper-parameters
    parser.add_argument("--rl_batch_size", type=int, default=10, help="batch size to sample architectures and compute rewards")
    parser.add_argument("--rl_update_per_epoch", action="store_true", help="whether to update architecture parameters per epoch")
    parser.add_argument("--rl_update_steps_per_epoch", type=int, default=300, help="number of update steps per epoch for architecture parameters")
    parser.add_argument("--rl_baseline_decay_weight", type=float, default=0.99, help="baseline decay weight for RL")
    parser.add_argument("--rl_tradeoff_ratio", type=float, default=0.1, help="tradeoff ratio for RL")

    # Archived unused search-stage forgetting arguments (kept here as notes only):
    # --search_cl_ortho_method
    # --search_cl_ortho_scale
    # --search_ortho_samples_per_task
    # --search_cl_kd_method
    # --search_cl_kd_logit_lambda
    # --search_cl_kd_temperature
    # --search_cl_kd_conf_threshold
    # --search_ewc_lambda
    # --search_ewc_samples_per_task
    # --search_ewc_online_interval
    # --search_cl_reg_method
    # --search_cl_reg_decay
    # --search_cl_reg_clip
    # --search_cl_penalty_clip
    # --search_replay_mode
    # --search_replay_capacity
    # --search_replay_capacity_ratio
    # --search_replay_per_batch
    # --search_replay_old_task_scale
    # --search_replay_old_task_scale_by_F
    # --arch_replay_lambda
    # --kd_lambda
    # --kd_temperature
    # --reg_lambda
    # --reg_use_ewc
    # They were used by the old "search stage + forgetting mitigation" pipeline.
    # The current NAS stage is intentionally clean and no longer consumes them.
    #
    # Archived unused retrain-stage forgetting arguments after switching to
    # personalized no-aggregation retrain:
    # --ewc_lambda
    # --ewc_samples_per_task
    # --ewc_online_interval
    # --cl_reg_method
    # --cl_reg_decay
    # --cl_reg_clip
    # --cl_penalty_clip
    # --cl_ortho_method
    # --cl_ortho_scale
    # --ortho_samples_per_task
    # --enable_ewc
    # --enable_orthogonal_update
    # --retrain_ewc_lambda
    # --retrain_ewc_samples_per_task
    # --retrain_ewc_online_interval
    # --retrain_cl_reg_method
    # --retrain_cl_reg_decay
    # --retrain_cl_reg_clip
    # --retrain_cl_penalty_clip
    # --retrain_cl_ortho_method
    # --retrain_cl_ortho_scale
    # --retrain_ortho_samples_per_task
    # EWC / orthogonal update are no longer part of the active retrain path.

    args = parser.parse_args()
    print("set格式化参数结束...")

    # 记录可被阶段覆盖的参数初值，便于超网/重训使用不同超参
    args._phase_base_params = {
        k: getattr(args, k)
        for k in [
            "cl_kd_method",
            "cl_kd_logit_lambda",
            "cl_kd_temperature",
            "cl_kd_conf_threshold",
            "enable_kd",
            "replay_mode",
            "replay_capacity",
            "replay_capacity_ratio",
            "replay_per_batch",
            "replay_old_task_scale",
            "replay_old_task_scale_by_F",
        ]
    }

    # 派生参数：保持原逻辑
    args.n_epochs = args.local_epoch_number * (args.last_round - args.start_round)

    # 构建保存目录名；若用户显式传入 --path，则直接使用该路径作为实验基路径。
    if args.path == parser.get_default("path"):
        base_path = "./output_test1/fednas-" + args.arch_algo + str(args.manual_seed)
        if args.target_hardware is not None:
            base_path += args.target_hardware
        args.path = base_path + str(args.n_cell_stages)

    # env_dir 命名（与原等价）
    args.save_env = "env_dir/search-{}-{}-{}-{}".format(
        args.arch_algo, args.train_batch_size, args.target_hardware, time.strftime("%Y%m%d-%H%M%S")
    )
    return args

# ----------------------------- 主流程 -----------------------------
def main():
    args = parse_args()

    # 阶段参数覆盖
    def _apply_phase_overrides(phase: str):
        base = getattr(args, "_phase_base_params", {})
        prefix = f"{phase}_"
        for k, base_val in base.items():
            override_val = getattr(args, prefix + k, None)
            setattr(args, k, base_val if override_val is None else override_val)
            
    def _attach_replay_cfg(run_cfg, a):
        """将回放相关超参挂到 run_config 上，便于 RunManager 读取；支持按全训练集比例设定容量。"""
        for k in ["replay_mode", "replay_capacity", "replay_per_batch", "replay_old_task_scale", "replay_old_task_scale_by_F"]:
            setattr(run_cfg, k, getattr(a, k, None))
        ratio = getattr(a, "replay_capacity_ratio", None)
        if ratio is not None:
            try:
                # 以“全训练集”估算：对 CIFAR10/100 使用 50k，并按客户端数量近似均分；否则退回当前任务 * num_tasks
                ds_lower = str(getattr(run_cfg, "dataset", "")).lower()
                if "cifar" in ds_lower:
                    total = 45000  # CIFAR-10/100 训练集规模
                    num_clients = max(1, int(getattr(run_cfg, "num_clients", 1)))
                    base = total // num_clients
                else:
                    base = run_cfg.data_provider.trn_set_length * getattr(run_cfg, "num_tasks", 1)
                cap = int(max(0, base * ratio))
                run_cfg.replay_capacity = cap
                print(f"[Replay] set capacity by ratio={ratio} -> {cap} (total_train≈{base})")
            except Exception as e:
                print(f"[Replay] failed to apply replay_capacity_ratio={ratio}: {e}")

    # 创建实验环境目录；保持异常可见
    try:
        create_exp_dir(args.save_env, scripts_to_save=glob.glob("*.py"))
    except Exception as e:
        print(line_info())
        print("出现异常：", e)

    # cuDNN 设定：沿用原值
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   # 设置程序可见的gpu环境变量

    # 随机种子
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    try:
        pynvml.nvmlInit()
        # 获取当前进程使用的 GPU 的 UUID（逻辑 GPU 0）
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        print(">>> 实际使用的物理 GPU UUID:", uuid)
    except Exception as e:
        # NVML 在共享/异常驱动环境下可能失败，不应阻断训练主流程
        print(f">>> NVML unavailable, skip UUID probing: {e}")
        try:
            print(f">>> torch visible cuda count: {torch.cuda.device_count()}")
        except Exception:
            pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    # 跨任务共享的 replay buffer（按 client 索引），避免每个任务重建导致忘记旧样本
    replay_buffers_across_tasks = [None for _ in range(args.num_users)]
    client_histories = [ClientHistory() for _ in range(args.num_users)]

    def _load_replay_buffers(task_path: str):
        """从上一任务目录加载 replay buffer 状态，供断点续跑/跨任务继承。"""
        if not args.enable_replay:
            return
        buf_path = os.path.join(task_path, "replay_buffers.pt")
        if not os.path.isfile(buf_path):
            return
        try:
            states = torch.load(buf_path, map_location="cpu")
            if isinstance(states, list):
                for idx, st in enumerate(states):
                    if st is None:
                        continue
                    buf = SimpleReplayBuffer(capacity=st.get("capacity", 0))
                    buf.load_state(st)
                    replay_buffers_across_tasks[idx] = buf
                print(f"[Replay] Loaded replay buffers from {buf_path}")
        except Exception as e:
            print(f"[Replay] Failed to load replay buffers from {buf_path}: {e}")

    def _save_replay_buffers(task_path: str):
        """将当前 replay buffer 状态保存到任务目录，便于下一任务/断点续跑。"""
        if not args.enable_replay:
            return 0.0
        buf_path = os.path.join(task_path, "replay_buffers.pt")
        states = []
        for buf in replay_buffers_across_tasks:
            states.append(buf.export_state() if buf is not None else None)
        start = time.time()
        try:
            atomic_torch_save(states, buf_path)
            print(f"[Replay] Saved replay buffers to {buf_path}")
        except Exception as e:
            print(f"[Replay] Failed to save replay buffers to {buf_path}: {e}")
        return max(0.0, time.time() - start)

    def _load_client_history_bundle(task_path: str):
        if not getattr(args, "enable_arch_prior", False):
            return False
        history_path = os.path.join(task_path, "client_arch_prior_histories.pt")
        if not os.path.isfile(history_path):
            return False
        loaded_histories = load_client_histories(history_path, args.num_users)
        for idx in range(args.num_users):
            client_histories[idx] = loaded_histories[idx]
        print(f"[ArchPrior] Loaded client histories from {history_path}")
        return True

    def _load_model_from_checkpoint(model, checkpoint_path: str, desc: str):
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                f"[{desc}] Loaded state_dict from {checkpoint_path}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            return True
        except Exception as e:
            print(f"[{desc}] Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def _restore_retrain_bootstrap(checkpoint_path: str, global_run_manager: RunManager, clients: list):
        if not checkpoint_path:
            return False
        if os.path.isdir(checkpoint_path):
            loaded_rounds = []
            restored_any = False
            for idx, rm in enumerate(clients):
                client_ckpt = os.path.join(checkpoint_path, f"client_{idx}", "checkpoint", "checkpoint.pth.tar")
                if not os.path.isfile(client_ckpt):
                    continue
                try:
                    checkpoint = torch.load(client_ckpt, map_location="cpu")
                    state_dict = checkpoint.get("state_dict", checkpoint)
                    rm.net.module.load_state_dict(state_dict, strict=False)
                    rm.round = int(checkpoint.get("round", -1)) + 1
                    opt_key = f"{idx}_weight_optimizer"
                    if opt_key in checkpoint:
                        rm.optimizer.load_state_dict(checkpoint[opt_key])
                    loaded_rounds.append(rm.round)
                    restored_any = True
                except Exception as e:
                    print(f"[AutoResume] Failed to restore client {idx} bootstrap from {client_ckpt}: {e}")
            if restored_any:
                global_run_manager.round = max(loaded_rounds) if loaded_rounds else 0
                print(f"[AutoResume] Bootstrapped personalized retrain stage from {checkpoint_path}")
                return True
            return False
        if not os.path.isfile(checkpoint_path):
            return False
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            print(f"[AutoResume] Failed to read retrain bootstrap checkpoint {checkpoint_path}: {e}")
            return False

        state_dict = checkpoint.get("state_dict", checkpoint)
        global_run_manager.net.module.load_state_dict(state_dict, strict=False)
        global_run_manager.round = 0
        for idx, rm in enumerate(clients):
            rm.net.module.load_state_dict(state_dict, strict=False)
            rm.round = 0
            opt_key = f"{idx}_weight_optimizer"
            if opt_key in checkpoint:
                try:
                    rm.optimizer.load_state_dict(checkpoint[opt_key])
                except Exception as e:
                    print(f"[AutoResume] Failed to load client {idx} optimizer from {checkpoint_path}: {e}")
        print(f"[AutoResume] Bootstrapped retrain stage from {checkpoint_path}")
        return True

    def _load_teacher_snapshot_like(model_template: torch.nn.Module, snapshot_path: str, desc: str):
        if snapshot_path is None:
            return None
        teacher = copy.deepcopy(model_template)
        if _load_model_from_checkpoint(teacher, snapshot_path, desc=desc):
            return teacher
        return None

    retrain_helpers = RetrainPipelineHelpers(
        attach_replay_cfg=_attach_replay_cfg,
        load_prev_task=load_prev_task,
        save_replay_buffers=_save_replay_buffers,
        load_model_from_checkpoint=_load_model_from_checkpoint,
        restore_retrain_bootstrap=_restore_retrain_bootstrap,
        load_teacher_snapshot_like=_load_teacher_snapshot_like,
        model_signature=model_signature,
        train_personalized_subnet=train_personalized_subnet,
    )

    # 遍历所有任务
    base_task_path = args.path
    auto_resume_manager = AutoResumeManager(base_task_path)
    runtime_context = RuntimeContext()
    args.runtime_context = runtime_context
    user_resume = bool(args.resume)
    user_skip_warmup = bool(args.skip_warmup)
    user_skip_search = bool(args.skip_search)
    auto_resume_plan = None
    effective_end_task_id = args.num_tasks if args.end_task_id is None else int(args.end_task_id)
    if effective_end_task_id < 1 or effective_end_task_id > int(args.num_tasks):
        raise ValueError(f"end_task_id must be in [1, {args.num_tasks}], got {effective_end_task_id}")
    if int(args.start_task_id) < 1 or int(args.start_task_id) > int(args.num_tasks):
        raise ValueError(f"start_task_id must be in [1, {args.num_tasks}], got {args.start_task_id}")
    if int(args.start_task_id) > effective_end_task_id:
        raise ValueError(
            f"start_task_id ({args.start_task_id}) must be <= end_task_id ({effective_end_task_id})"
        )
    if args.auto_resume:
        auto_resume_plan = auto_resume_manager.resolve(args.num_tasks, args.object_to_search)
        runtime_context.auto_resume_plan = copy.deepcopy(auto_resume_plan)
        if auto_resume_plan.get("all_completed"):
            print("[AutoResume] 所有任务都已完成，无需继续。")
            return
        args.start_task_id = int(auto_resume_plan["task_id"])
        if args.start_task_id > effective_end_task_id:
            print(
                f"[AutoResume] 恢复起点 task {args.start_task_id} 超出本次执行上限 task {effective_end_task_id}，无需继续。"
            )
            return
        print(
            f"[AutoResume] 计划从 task {args.start_task_id} 恢复，"
            f"phase={auto_resume_plan.get('phase')} resume={auto_resume_plan.get('resume', False)}"
        )

    for task_id in range(args.start_task_id, effective_end_task_id + 1):
        print(f"开始执行任务 {task_id} (本次范围: {args.start_task_id}-{effective_end_task_id}, 总任务数: {args.num_tasks})")
        args.task_id = task_id  # 设置当前任务的 task_id
        args.search = True
        args.resume = user_resume
        args.skip_warmup = user_skip_warmup
        args.skip_search = user_skip_search
        runtime_context.reset_for_task()

        if auto_resume_plan is not None and task_id == int(auto_resume_plan["task_id"]):
            args.resume = bool(auto_resume_plan.get("resume", False))
            args.skip_warmup = bool(auto_resume_plan.get("skip_warmup", args.skip_warmup))
            args.skip_search = bool(auto_resume_plan.get("skip_search", args.skip_search))
            runtime_context.set_resume_task_plan(auto_resume_plan)
            print(
                f"[AutoResume] task{task_id} 使用自动恢复配置: "
                f"skip_warmup={args.skip_warmup}, skip_search={args.skip_search}, resume={args.resume}"
            )
        elif auto_resume_plan is not None and task_id > int(auto_resume_plan["task_id"]):
            # 仅对首个恢复任务特殊处理，后续任务恢复默认行为
            auto_resume_plan = None

        # 每个任务开始时按执行模式恢复参数配置
        if args.retrain_only:
            _apply_phase_overrides("retrain")
            args.search = False
            args.skip_warmup = True
            args.skip_search = True
        else:
            _apply_phase_overrides("search")

        args.path = base_task_path + f"-task{task_id}"  # 每个任务使用不同的保存路径
        os.makedirs(args.path, exist_ok=True)
        auto_resume_manager.handle_event(task_id, "task_started", object_to_search=args.object_to_search)
        # 记录命令行，便于复现实验
        try:
            cmd_path = os.path.join(args.path, "command.txt")
            with open(cmd_path, "w") as fout:
                fout.write(" ".join(sys.argv))
        except Exception:
            pass
        # 尝试继承上一任务的超网权重
        prev_task_path = base_task_path + f"-task{task_id - 1}"
        if getattr(args, "enable_arch_prior", False):
            loaded_current_history = _load_client_history_bundle(args.path)
            if (not loaded_current_history) and task_id > 1:
                _load_client_history_bundle(prev_task_path)
        # 断点续跑 / 非首任务：尝试加载上一任务的 replay buffer
        if (
            args.enable_replay
            and replay_buffers_across_tasks.count(None) == len(replay_buffers_across_tasks)
        ):
            resume_replay_path = None
            if runtime_context.resume_task_plan is not None:
                resume_replay_path = runtime_context.resume_task_plan.get("replay_buffer_path")
            if resume_replay_path is not None:
                _load_replay_buffers(os.path.dirname(resume_replay_path))
            elif task_id > 1:
                _load_replay_buffers(prev_task_path)

        # 组装 run_config
        args.lr_schedule_param = None
        args.opt_param = {"momentum": args.momentum, "nesterov": not args.no_nesterov}

        clients_run_config_arr = []
        for idx in range(args.num_users):
            args.client_id = idx
            run_cfg = CifarRunConfig(**args.__dict__ , is_client = True)
            _attach_replay_cfg(run_cfg, args)
            clients_run_config_arr.append(run_cfg)

        if args.retrain_only:
            print("-----------------------------------------------------retrain only: reuse learned subnets and rerun personalized retrain-----------------------------------------------------")
            current_task_path = args.path
            learned_net_path = os.path.join(current_task_path, "learned_net")
            if args.object_to_search == "supernet":
                net_config_path = os.path.join(learned_net_path, "net.config")
                if not os.path.isfile(net_config_path):
                    raise FileNotFoundError(
                        f"retrain_only requires existing learned_net/net.config under {learned_net_path}"
                    )
                run_supernet_retrain_pipeline(
                    args=args,
                    task_id=task_id,
                    prev_task_path=prev_task_path,
                    current_task_path=current_task_path,
                    global_server=None,
                    clients_run_config_arr=clients_run_config_arr,
                    replay_buffers_across_tasks=replay_buffers_across_tasks,
                    runtime_context=runtime_context,
                    auto_resume_manager=auto_resume_manager,
                    user_resume=user_resume,
                    helpers=retrain_helpers,
                )
            elif args.object_to_search == "baseline":
                run_baseline_retrain_pipeline(
                    args=args,
                    task_id=task_id,
                    prev_task_path=prev_task_path,
                    current_task_path=current_task_path,
                    replay_buffers_across_tasks=replay_buffers_across_tasks,
                    runtime_context=runtime_context,
                    auto_resume_manager=auto_resume_manager,
                    helpers=retrain_helpers,
                )
            else:
                raise NotImplementedError(f"retrain_only does not support object_to_search={args.object_to_search}")
            auto_resume_manager.handle_event(task_id, "task_completed", path=args.path)
            continue

        previous_subnet = None
        if task_id > 1:
            previous_subnet = load_previous_subnet_for_evaluation(
                os.path.join(prev_task_path, "learned_net")
            )
        quick_eval_metrics = []
        for run_cfg in clients_run_config_arr:
            metric = quick_evaluate_previous_subnet(previous_subnet, run_cfg, max_batches=1)
            if metric is not None:
                quick_eval_metrics.append(metric)
        if quick_eval_metrics:
            avg_loss = sum(item["loss"] for item in quick_eval_metrics) / len(quick_eval_metrics)
            avg_top1 = sum(item["top1"] for item in quick_eval_metrics) / len(quick_eval_metrics)
            print(
                f"[QuickEval] task{task_id} previous_subnet_on_current_task "
                f"clients={len(quick_eval_metrics)} loss={avg_loss:.4f} top1={avg_top1:.2f}"
            )

        # 解析网络结构相关字符串参数
        def _ensure_int_list(x):
            if isinstance(x, (list, tuple)):
                return [int(v) for v in x]
            if isinstance(x, str):
                s = x.strip()
                if s == "":
                    return []
                return [int(v) for v in s.split(",")]
            # 兼容单个 int 或其他可转 int 的类型
            try:
                return [int(x)]
            except Exception:
                raise ValueError(f"无法解析为 int 列表: {x}")

        args.width_stages = _ensure_int_list(args.width_stages)
        args.n_cell_stages = _ensure_int_list(args.n_cell_stages)
        args.stride_stages = _ensure_int_list(args.stride_stages)
        
        args.conv_candidates = [
            # 'ResNetBlock','DenseNetBlock','SEBlock',
            '3x3_MBConv1', '3x3_MBConv2', '3x3_MBConv3', '3x3_MBConv4', '3x3_MBConv5', '3x3_MBConv6',
            '5x5_MBConv1', '5x5_MBConv2', '5x5_MBConv3', '5x5_MBConv4', '5x5_MBConv5', '5x5_MBConv6',
            '7x7_MBConv1', '7x7_MBConv2', '7x7_MBConv3', '7x7_MBConv4', '7x7_MBConv5', '7x7_MBConv6'
        ]

        # ---------------- SuperNet 初始化 ----------------
        super_net = SuperProxylessNASNets(
            width_stages=args.width_stages,
            n_cell_stages=args.n_cell_stages,
            stride_stages=args.stride_stages,
            conv_candidates=args.conv_candidates,
            n_classes=clients_run_config_arr[0].data_provider.n_classes,    # 类别都是依据cifar来返回10 / 100
            width_mult=args.width_mult,
            bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=args.dropout,
            inference_device=args.object_to_search,
        )
        
        # 如果非首任务，尝试加载上一任务的权重
        loaded_prev_supernet = False
        loaded_task1_bootstrap = False
        task1_bootstrap_ckpt_path = None
        if task_id == 1:
            task1_bootstrap_ckpt_path = os.path.abspath(getattr(args, "task1_bootstrap_ckpt", "./global.pth.tar"))
            loaded_task1_bootstrap = load_task1_bootstrap(super_net, task1_bootstrap_ckpt_path)
        if task_id > 1:
            loaded_prev_supernet = load_prev_task(super_net, prev_task_path)
        inherited_supernet = loaded_prev_supernet or loaded_task1_bootstrap
            
        # 记录当前超网签名，便于判断是否继承成功（与上一任务相比应保持连续，不应回到随机分布）
        log_dir = os.path.join(args.path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        sig_path = os.path.join(log_dir, f"inherit_check_task{task_id}.log")
        
        with open(sig_path, "a") as fout:
            fout.write(json.dumps({
                "stage": "init_supernet",
                "task_id": task_id,
                "loaded_prev_supernet": loaded_prev_supernet,
                "loaded_task1_bootstrap": loaded_task1_bootstrap,
                "task1_bootstrap_ckpt_path": task1_bootstrap_ckpt_path,
                "signature": model_signature(super_net),
            }) + "\n")
        base_supernet_state = copy.deepcopy(super_net.state_dict())
        
        # ---------------- 搜索配置 ----------------
        if args.arch_opt_type == "adam":
            args.arch_opt_param = {"betas": (args.arch_adam_beta1, args.arch_adam_beta2),
                                   "eps": args.arch_adam_eps}
        else:
            args.arch_opt_param = None

        args.ref_value = None if args.target_hardware is None else \
            REF_VALUES[args.target_hardware]["%.2f" % args.width_mult]

        if args.arch_algo == "grad":
            # grad 正则化参数
            if args.grad_reg_loss_type == "add#linear":
                args.grad_reg_loss_params = {"lambda": args.grad_reg_loss_lambda}
            elif args.grad_reg_loss_type == "mul#log":
                args.grad_reg_loss_params = {"alpha": args.grad_reg_loss_alpha,
                                             "beta": args.grad_reg_loss_beta}
            else:
                args.grad_reg_loss_params = None

            arch_search_config = GradientArchSearchConfig(**args.__dict__)
        elif args.arch_algo == "rl":
            arch_search_config = RLArchSearchConfig(**args.__dict__)
        else:
            raise NotImplementedError

        # ---------------- 全局 server ----------------
        args.client_id = 0          # 任意给 id 即可
        run_config_global_server = CifarRunConfig(**args.__dict__, is_client = False)
        _attach_replay_cfg(run_config_global_server, args)
        arch_search_config_global_server = copy.deepcopy(arch_search_config)
        global_server = ArchSearchRunManager(
            args.path, super_net, run_config_global_server,
            arch_search_config_global_server, warmup=args.warmup, task_id=args.task_id,
            init_model=not inherited_supernet,
            replay_buffer=None,
        )
        # 继承上一任务的全局优化器状态（在 global_server 创建后再尝试）
        if task_id > 1:
            try:
                load_prev_task_optimizers(global_server, prev_task_path, client_id=0, task_id_from=task_id - 1, is_server=True)
            except Exception as e:
                print(f"[FCL] Failed to load prev global optimizers: {e}")
        
        print("The global_server user has {} training data, {} valid data and {} test data.".format(
            global_server.run_manager.run_config.data_provider.trn_set_length,
            global_server.run_manager.run_config.data_provider.val_set_length,
            global_server.run_manager.run_config.data_provider.tst_set_length,
        ))

        # ---------------- Client 集群 ----------------
        clients = []
        mobile_client_idx_arr, cpu_client_idx_arr = [], []
        gpu8_client_idx_arr, flops_client_idx_arr = [], []
        none_client_idx_arr, all_client_idx_arr = [], []

        for idx in range(args.num_users):
            # 打印当前配置类型
            if isinstance(arch_search_config, RLArchSearchConfig):
                print(line_info()); print("RLArchSearchConfig")
            elif isinstance(arch_search_config, GradientArchSearchConfig):
                print(line_info()); print("GradientArchSearchConfig")
            else:
                print(line_info()); print("Can not know arch_search_config!")

            # 每个 client 独立的搜索配置（含 target_hardware、ref_value）
            asc_local = copy.deepcopy(arch_search_config)
            asc_local.target_hardware = set_target_hardware(idx=idx)
            asc_local.ref_value = REF_VALUES[asc_local.target_hardware]["%.2f" % args.width_mult] \
                if asc_local.target_hardware is not None else None

            # 记录不同类型 client 的索引
            all_client_idx_arr.append(idx)
            if   asc_local.target_hardware == "mobile": mobile_client_idx_arr.append(idx)
            elif asc_local.target_hardware == "cpu":    cpu_client_idx_arr.append(idx)
            elif asc_local.target_hardware == "gpu8":   gpu8_client_idx_arr.append(idx)
            elif asc_local.target_hardware == "flops":  flops_client_idx_arr.append(idx)
            else:                                       none_client_idx_arr.append(idx)

            # 为每个 client 构建独立 supernet 与运行管理器
            local_client_super_net = SuperProxylessNASNets(
                width_stages=args.width_stages, n_cell_stages=args.n_cell_stages,
                stride_stages=args.stride_stages, conv_candidates=args.conv_candidates,
                n_classes=clients_run_config_arr[idx].data_provider.n_classes,
                width_mult=args.width_mult, bn_param=(args.bn_momentum, args.bn_eps),
                dropout_rate=args.dropout, inference_device=asc_local.target_hardware,
            )
            # 使用上一任务的初始化
            local_client_super_net.load_state_dict(base_supernet_state, strict=False)
            client = ArchSearchRunManager(
                args.path, local_client_super_net,
                clients_run_config_arr[idx], asc_local, task_id=args.task_id,
                init_model=not inherited_supernet,
                replay_buffer=None,
            )
            _attach_replay_cfg(client.run_manager.run_config, args)
            clients.append(client)

            print("{} client has {} training data, {} valid data and {} test data.".format(
                client.run_manager.run_config.data_provider.client_id,
                client.run_manager.run_config.data_provider.trn_set_length,
                client.run_manager.run_config.data_provider.val_set_length,
                client.run_manager.run_config.data_provider.tst_set_length,
            ))
            # 继承上一任务的优化器动量等状态，便于连续任务平滑过渡
            if task_id > 1:
                load_prev_task_optimizers(client, prev_task_path, client_id=idx, task_id_from=task_id - 1)

        # ---------------- 训练阶段分支 ----------------
        super_server = global_server
        # 封装剪枝/推理场景，减少后续重复代码
        def _run_pruning_case(target_hw, idx_arr, msg):
            print(line_info()); print(msg)
            cm = ClusteringMachine(
                target_hardware=target_hw,
                config=args,
                global_server=super_server,
                clients_idx_arr=idx_arr,
                clients=clients,
                start_round=args.start_round,
                last_round=args.last_round,
                path=args.path,
                task_id=task_id,
            )
            cm.run()
            cm.test_inference()

        if args.object_to_search == "supernet":
            def _search_progress_callback(event, **payload):
                auto_resume_manager.handle_event(task_id, event, **payload)
                return 0.0

            runtime_context.set_progress_callback(_search_progress_callback)
            if args.skip_search:
                print("-----------------------------------------------------skip search: directly retrain learned_net-----------------------------------------------------")
            else:
                print("-----------------------------------------------------case1: training super_net-----------------------------------------------------")
                super_server = run_nas_search_for_task(
                    args,
                    task_id,
                    global_server,
                    clients,
                    all_client_idx_arr,
                    prev_task_path=prev_task_path,
                    current_task_path=args.path,
                    client_histories=client_histories,
                )
                print("完成super_net训练阶段")

                super_server.load_model()
                super_server.get_normal_net()
                auto_resume_manager.handle_event(
                    task_id,
                    "learned_net_ready",
                    learned_net_path=os.path.join(args.path, "learned_net"),
                )
                print('获取固化网络成功')
                if args.auto_cleanup_warmup_ckpt:
                    _cleanup_search_checkpoint_files(args.path, remove_global=False, remove_warmup=True)
                if args.auto_cleanup_prev_task_search_ckpt and prev_task_path:
                    _cleanup_search_checkpoint_files(prev_task_path, remove_global=True, remove_warmup=True)
            
            runtime_context.clear_progress_callback()
            
            # 三、子网重训阶段
            _apply_phase_overrides("retrain")
            current_task_path = args.path
            run_supernet_retrain_pipeline(
                args=args,
                task_id=task_id,
                prev_task_path=prev_task_path,
                current_task_path=current_task_path,
                global_server=super_server,
                clients_run_config_arr=clients_run_config_arr,
                replay_buffers_across_tasks=replay_buffers_across_tasks,
                runtime_context=runtime_context,
                auto_resume_manager=auto_resume_manager,
                user_resume=user_resume,
                helpers=retrain_helpers,
            )


        elif args.object_to_search == "baseline":
            print(line_info()); print("-----------------------------case baseline: fixed backbone--------------------------------")
            _apply_phase_overrides("retrain")
            current_task_path = args.path
            run_baseline_retrain_pipeline(
                args=args,
                task_id=task_id,
                prev_task_path=prev_task_path,
                current_task_path=current_task_path,
                replay_buffers_across_tasks=replay_buffers_across_tasks,
                runtime_context=runtime_context,
                auto_resume_manager=auto_resume_manager,
                helpers=retrain_helpers,
            )


        elif args.object_to_search == "cpu":
            _run_pruning_case("cpu", cpu_client_idx_arr, "-----------------------------case2: pruning for cpu--------------------------------")

        elif args.object_to_search == "gpu8":
            _run_pruning_case("gpu8", gpu8_client_idx_arr, "---------------------------------case3: pruning for gpu8------------------------------------------")

        elif args.object_to_search == "flops":
            _run_pruning_case("flops", flops_client_idx_arr, "------------------------case4: pruning for flops--------------------------------------")

        elif args.object_to_search is None:
            _run_pruning_case(None, none_client_idx_arr, "--------------------------------------case5: pruning for None------------------------------------------")

        auto_resume_manager.handle_event(task_id, "task_completed")
        print(f"任务 {task_id}/{args.num_tasks} 完成！")

if __name__ == "__main__":
    main()
