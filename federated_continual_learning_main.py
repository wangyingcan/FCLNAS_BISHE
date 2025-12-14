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

from clustering_machine import *
from data_providers.cifar100_fcl_dirichlet_split import CifarDataProvider100
from nas_manager import ArchSearchRunManager, GradientArchSearchConfig, RLArchSearchConfig
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
        return
    try:
        torch.save(state, path)
        print(f"[{desc}] Saved state to {path}")
    except Exception as e:
        print(f"[{desc}] Failed to save state to {path}: {e}")

# ----------------------------- 参数解析与派生 -----------------------------
def parse_args() -> argparse.Namespace:
    print("set格式化参数开始...")
    parser = argparse.ArgumentParser()

    # Federated Learning
    parser.add_argument("--gpu", default="0,1,2,3", help="gpu available to use")
    parser.add_argument("--num_users", type=int, default=10, help="number of clients: K")
    parser.add_argument("--num_tasks", type=int, default=10, help="number of tasks: K")
    parser.add_argument("--start_task_id", type=int, default=1, help="start task id")
    parser.add_argument("--object_to_search", type=str, default="supernet",
                        choices=["supernet", "cpu", "gpu8", "flops", "baseline"],
                        help="search target(0:supernet, 1:cpu, 2:gpu8, 3:flops) latency for 1/2 , compute source for 3.")
    parser.add_argument("--iid", type=int, default=0,
                        help="Default set 1 to IID. Set to 0 for non-IID.")
    parser.add_argument("--unequal", type=int, default=1,
                        help="whether to use unequal data splits for non-i.i.d setting (use 0 for equal splits)")

    # ProxylessNAS
    parser.add_argument("--warmup", action="store_true", help="if have not warmup(only pretrain supernet, don't nas), please set it True")
    parser.add_argument("--path", type=str, default="./output/proxyless-", help="checkpoint save path")
    parser.add_argument("--save_env", type=str, default="EXP", help="experiment time name to save exp code")
    parser.add_argument("--resume", action="store_true", help="load last checkpoint")
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed to make experiments reproducible")
    parser.add_argument("--start_round", default=0, type=int, help="start round in fed_search")
    parser.add_argument("--last_round", default=10, type=int, help="last round in fed_search. 125 for all clients. 175 for cpu/gpu.")  # 讨论硬件差异时就多训练些联邦轮次
    parser.add_argument("--retrain_start_round", type=int, default=0, help="重训阶段起始轮次，默认沿用 start_round")
    parser.add_argument("--retrain_last_round", type=int, default=20, help="重训阶段最后轮次，默认沿用 last_round")
    parser.add_argument("--retrain_sequence_from_task1", action="store_true",
                        help="重训阶段依次从 task1 训练到当前任务，每个任务各跑 retrain_last_round 轮")
    
    
    parser.add_argument("--ewc_lambda", type=float, default=0.0,
                        help="重训阶段的 EWC 正则强度，0 表示关闭")
    parser.add_argument("--ewc_samples_per_task", type=int, default=512,
                        help="计算 Fisher 时使用的样本上限，设为 0 可跳过 Fisher 估计")
    parser.add_argument("--ewc_online_interval", type=int, default=0,
                        help="每隔多少 retrain round 在线累积一次 Fisher（0 表示仅在任务结束后计算）")
    parser.add_argument("--cl_reg_method", type=str, default="none",
                        choices=["ewc", "mas", "rwalk","none"],
                        help="持续学习正则方法：ewc / MAS / RWalk（默认 mas）")
    parser.add_argument("--cl_reg_decay", type=float, default=1.0,
                        help="重要性衰减系数，<1 表示按比例保留历史重要性后再累加新重要性")
    parser.add_argument("--cl_reg_clip", type=float, default=None,
                        help="重要性裁剪上限，None 表示不裁剪")
    parser.add_argument("--cl_penalty_clip", type=float, default=None,
                        help="正则项裁剪上限，防止 total_loss 爆炸，None 表示不裁剪")
    parser.add_argument("--cl_kd_method", type=str, default="none",
                        choices=["none", "logit", "logit_conf"],
                        help="持续学习蒸馏方法：none / logit / logit_conf")
    parser.add_argument("--cl_kd_logit_lambda", type=float, default=0.0,
                        help="logit 蒸馏 loss 权重，0 表示关闭蒸馏")
    parser.add_argument("--cl_kd_temperature", type=float, default=2.0,
                        help="logit 蒸馏的温度系数 T")
    parser.add_argument("--cl_kd_conf_threshold", type=float, default=0.5,
                        help="logit_conf 模式下 teacher 置信度阈值，低于该值的类不参与 KD")
    parser.add_argument("--cl_ortho_method",type=str,default="none",choices=["none", "ogd", "pcgrad", "kd_ortho","prev_grad_ortho","kd_prev_grad_ortho"],help="持续学习的正交更新方法：none / ogd / pcgrad / kd_ortho",)
    parser.add_argument("--cl_ortho_scale",type=float,default=1.0,help="正交投影强度系数，1.0 表示完全投影到正交子空间，<1 为部分投影",)
    parser.add_argument("--ortho_samples_per_task",type=int,default=2048,help="估计旧任务梯度方向时使用的样本上限，0 表示不估计（禁用正交基更新）",)
    parser.add_argument("--replay_mode", type=str, default="none", choices=["none", "global", "task_balanced", "age_priority"], help="experience replay 模式：none/global/task_balanced/age_priority")
    parser.add_argument("--replay_capacity", type=int, default=0, help="重放缓冲区最大样本量（全局计数）")
    parser.add_argument("--replay_capacity_ratio", type=float, default=None, help="按全训练集样本数的比例设置缓冲区容量（0~1），高于 replay_capacity 时覆盖之")
    parser.add_argument("--replay_per_batch", type=int, default=0, help="每个 batch 从缓冲区重放的样本数")
    parser.add_argument("--replay_old_task_scale", type=float, default=1.0, help="age_priority 模式下旧任务样本的权重缩放，>1 让旧任务更容易被采样")
    parser.add_argument("--replay_old_task_scale_by_F", type=float, default=0.0, help="按遗忘程度动态放大旧任务样本权重，0 表示不启用，单位：每个遗忘点数的放大系数")
    
    
    # 分阶段可选覆盖：search_* 用于超网预热/搜索，retrain_* 用于重训；未提供则回落到上述全局参数
    parser.add_argument("--search_cl_ortho_method", type=str, default=None, choices=["none","ogd","pcgrad","kd_ortho","prev_grad_ortho","kd_prev_grad_ortho"])
    parser.add_argument("--search_cl_ortho_scale", type=float, default=None)
    parser.add_argument("--search_ortho_samples_per_task", type=int, default=None)
    parser.add_argument("--search_cl_kd_method", type=str, default=None, choices=["none","logit","logit_conf"])
    parser.add_argument("--search_cl_kd_logit_lambda", type=float, default=None)
    parser.add_argument("--search_cl_kd_temperature", type=float, default=None)
    parser.add_argument("--search_cl_kd_conf_threshold", type=float, default=None)
    parser.add_argument("--search_ewc_lambda", type=float, default=None)
    parser.add_argument("--search_ewc_samples_per_task", type=int, default=None)
    parser.add_argument("--search_ewc_online_interval", type=int, default=None)
    parser.add_argument("--search_cl_reg_method", type=str, default=None, choices=["ewc","mas","rwalk"])
    parser.add_argument("--search_cl_reg_decay", type=float, default=None)
    parser.add_argument("--search_cl_reg_clip", type=float, default=None)
    parser.add_argument("--search_cl_penalty_clip", type=float, default=None)
    parser.add_argument("--search_replay_mode", type=str, default=None, choices=["none", "global", "task_balanced", "age_priority"])
    parser.add_argument("--search_replay_capacity", type=int, default=None)
    parser.add_argument("--search_replay_capacity_ratio", type=float, default=None)
    parser.add_argument("--search_replay_per_batch", type=int, default=None)
    parser.add_argument("--search_replay_old_task_scale", type=float, default=None)
    parser.add_argument("--search_replay_old_task_scale_by_F", type=float, default=None)
    
    parser.add_argument("--retrain_cl_ortho_method", type=str, default=None, choices=["none","ogd","pcgrad","kd_ortho","prev_grad_ortho","kd_prev_grad_ortho"])
    parser.add_argument("--retrain_cl_ortho_scale", type=float, default=None)
    parser.add_argument("--retrain_ortho_samples_per_task", type=int, default=None)
    parser.add_argument("--retrain_cl_kd_method", type=str, default=None, choices=["none","logit","logit_conf"])
    parser.add_argument("--retrain_cl_kd_logit_lambda", type=float, default=None)
    parser.add_argument("--retrain_cl_kd_temperature", type=float, default=None)
    parser.add_argument("--retrain_cl_kd_conf_threshold", type=float, default=None)
    parser.add_argument("--retrain_ewc_lambda", type=float, default=None)
    parser.add_argument("--retrain_ewc_samples_per_task", type=int, default=None)
    parser.add_argument("--retrain_ewc_online_interval", type=int, default=None)
    parser.add_argument("--retrain_cl_reg_method", type=str, default=None, choices=["ewc","mas","rwalk"])
    parser.add_argument("--retrain_cl_reg_decay", type=float, default=None)
    parser.add_argument("--retrain_cl_reg_clip", type=float, default=None)
    parser.add_argument("--retrain_cl_penalty_clip", type=float, default=None)
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
                        help="fixed backbone name in torchvision.models, e.g. resnet18/resnet34/resnet50")
    parser.add_argument("--baseline_pretrained", action="store_true",
                        help="use torchvision pretrained weights for baseline backbones")

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
    parser.add_argument("--arch_replay_lambda", type=float, default=0.0,
                        help="weight for replay loss when updating architecture parameters (gradient/RL search)")
    parser.add_argument("--kd_lambda", type=float, default=0.0,
                        help="distillation loss weight; 0 disables KD, 0.5 means CE/KD各占一半")
    parser.add_argument("--kd_temperature", type=float, default=2.0,
                        help="distillation温度，>1 会软化 logits，常见取值 2~4")
    parser.add_argument("--reg_lambda", type=float, default=1e-1,
                        help="L2 正则约束当前权重偏离上一轮 teacher，0 表示关闭")
    parser.add_argument("--reg_use_ewc", action="store_true",
                        help="开启简单的 EWC 风格：对锚点权重的偏移按 Fisher 近似加权")
    parser.add_argument("--skip_warmup", action="store_true",
                        help="跳过 warmup 阶段，直接进入 search，用于快速验证")
    parser.add_argument("--skip_search", action="store_true",
                        help="跳过 warmup/search，直接进入 learned_net 重训（要求已有 learned_net/init 和 net.config）")

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

    args = parser.parse_args()
    print("set格式化参数结束...")

    # 记录可被阶段覆盖的参数初值，便于超网/重训使用不同超参
    args._phase_base_params = {
        k: getattr(args, k)
        for k in [
            "cl_ortho_method",
            "cl_ortho_scale",
            "ortho_samples_per_task",
            "cl_kd_method",
            "cl_kd_logit_lambda",
            "cl_kd_temperature",
            "cl_kd_conf_threshold",
            "ewc_lambda",
            "ewc_samples_per_task",
            "ewc_online_interval",
            "cl_reg_method",
            "cl_reg_decay",
            "cl_reg_clip",
            "cl_penalty_clip",
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

    # 构建保存目录名（原文件两次赋值，合并为等价流程，行为不变）
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

    # 随机种子
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu   # 设置程序可见的gpu环境变量
    
    pynvml.nvmlInit()
    # 获取当前进程使用的 GPU 的 UUID
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 逻辑 GPU 0
    uuid = pynvml.nvmlDeviceGetUUID(handle)

    print(">>> 实际使用的物理 GPU UUID:", uuid)

    # 跨任务共享的 replay buffer（按 client 索引），避免每个任务重建导致忘记旧样本
    replay_buffers_across_tasks = [None for _ in range(args.num_users)]

    def _load_replay_buffers(task_path: str):
        """从上一任务目录加载 replay buffer 状态，供断点续跑/跨任务继承。"""
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
        buf_path = os.path.join(task_path, "replay_buffers.pt")
        states = []
        for buf in replay_buffers_across_tasks:
            states.append(buf.export_state() if buf is not None else None)
        try:
            torch.save(states, buf_path)
            print(f"[Replay] Saved replay buffers to {buf_path}")
        except Exception as e:
            print(f"[Replay] Failed to save replay buffers to {buf_path}: {e}")

    # 遍历所有任务
    base_task_path = args.path
    for task_id in range(args.start_task_id, args.num_tasks + 1):
        print(f"开始执行任务 {task_id}/{args.num_tasks}")
        args.task_id = task_id  # 设置当前任务的 task_id
        args.search = True

        args.path = base_task_path + f"-task{task_id}"  # 每个任务使用不同的保存路径
        os.makedirs(args.path, exist_ok=True)
        # 记录命令行，便于复现实验
        try:
            cmd_path = os.path.join(args.path, "command.txt")
            with open(cmd_path, "w") as fout:
                fout.write(" ".join(sys.argv))
        except Exception:
            pass
        # 尝试继承上一任务的超网权重
        prev_task_path = base_task_path + f"-task{task_id - 1}"
        # 断点续跑 / 非首任务：尝试加载上一任务的 replay buffer
        if args.start_task_id != 1 and replay_buffers_across_tasks.count(None) == len(replay_buffers_across_tasks):
            _load_replay_buffers(prev_task_path)

        # 组装 run_config
        args.lr_schedule_param = None
        args.opt_param = {"momentum": args.momentum, "nesterov": not args.no_nesterov}

        clients_run_config_arr = []
        for idx in range(args.num_users):
            args.client_id = idx
            clients_run_config_arr.append(CifarRunConfig(**args.__dict__ , is_client = True))

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

        # args.conv_candidates = [
        #     "3x3_MBConv2", "3x3_MBConv3", "3x3_MBConv4",
        #     "3x3_MBConv5", "3x3_MBConv6", "5x5_MBConv3",
        # ]
        
        args.conv_candidates = [
            'ResNetBlock','DenseNetBlock','SEBlock',
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
        if task_id > 1:
            loaded_prev_supernet = load_prev_task(super_net, prev_task_path)
            
        # 记录当前超网签名，便于判断是否继承成功（与上一任务相比应保持连续，不应回到随机分布）
        log_dir = os.path.join(args.path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        sig_path = os.path.join(log_dir, f"inherit_check_task{task_id}.log")
        
        with open(sig_path, "a") as fout:
            fout.write(json.dumps({
                "stage": "init_supernet",
                "task_id": task_id,
                "loaded_prev_supernet": loaded_prev_supernet,
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
            init_model=not loaded_prev_supernet,
            replay_buffer=replay_buffers_across_tasks[0] if replay_buffers_across_tasks else None,
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
                init_model=not loaded_prev_supernet,
                replay_buffer=replay_buffers_across_tasks[idx] if replay_buffers_across_tasks else None,
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
            # search 阶段可选覆盖
            _apply_phase_overrides("search")
            # teacher：上一任务固化子网，用于 supernet KD / kd_ortho
            super_teacher_model = None
            # 启用 KD / 正交 / 权重锚定(EWC) 任一项，都需要上一任务 teacher
            need_super_teacher = (
                args.cl_kd_logit_lambda > 0
                or args.cl_ortho_method == "kd_ortho"
                or args.reg_lambda > 0
                or args.reg_use_ewc
                or args.ewc_lambda > 0
            )
            if need_super_teacher and task_id > 1:
                try:
                    # Teacher 使用上一任务的超网 checkpoint（global/warmup），而非 learned 子网
                    super_teacher_model = SuperProxylessNASNets(
                        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages,
                        stride_stages=args.stride_stages, conv_candidates=args.conv_candidates,
                        n_classes=global_server.run_manager.run_config.data_provider.n_classes,
                        width_mult=args.width_mult, bn_param=(args.bn_momentum, args.bn_eps),
                        dropout_rate=args.dropout, inference_device="super_net",
                    )
                    loaded_teacher = load_prev_task(super_teacher_model, prev_task_path)
                    if not loaded_teacher:
                        super_teacher_model = None
                        print(f"[Supernet] 未能加载上一任务超网作为 teacher，KD/kd_ortho 将跳过")
                    else:
                        print(f"[Supernet] 成功加载上一任务{prev_task_path}超网作为 teacher")
                except Exception as e:
                    super_teacher_model = None
                    print(f"[Supernet] 加载上一任务超网 teacher 失败: {e}")

            # 一、超网训练阶段
            if args.skip_search:
                print("-----------------------------------------------------skip search: directly retrain learned_net-----------------------------------------------------")
            else:
                print("-----------------------------------------------------case1: training super_net-----------------------------------------------------")
                # 加载上一任务的 EWC/正交状态并广播到 server & clients，便于跨任务连续正则
                ewc_state_path = os.path.join(args.path, "ewc_state.pt")
                prev_ewc_state_path = os.path.join(prev_task_path, "ewc_state.pt")
                ortho_state_path = os.path.join(args.path, "ortho_state.pt")
                prev_ortho_state_path = os.path.join(prev_task_path, "ortho_state.pt")
                # 首任务且非 resume 时，不继承现有 ewc/ortho 文件，避免旧状态污染
                if task_id == 1 and not getattr(args, "resume", False):
                    ewc_state, ortho_state = None, None
                    print("[Supernet] fresh task1 run, skip loading existing ewc/ortho state")
                else:
                    ewc_state = _load_state_with_fallback(ewc_state_path, prev_ewc_state_path, desc="Supernet")
                    ortho_state = _load_state_with_fallback(ortho_state_path, prev_ortho_state_path, desc="Supernet")
                
                def _broadcast_state(ewc_s, ortho_s):
                    global_server.run_manager.load_ewc_state(ewc_s)
                    global_server.run_manager.load_ortho_state(ortho_s)
                    for cli in clients:
                        cli.run_manager.load_ewc_state(ewc_s)
                        cli.run_manager.load_ortho_state(ortho_s)
                        
                _broadcast_state(ewc_state, ortho_state)

                # 先下发 teacher 到 server/client 侧 run_manager，保证 super_cm 期间可用
                if super_teacher_model is not None:
                    global_server.run_manager.set_teacher(super_teacher_model)
                    for cli in clients:
                        cli.run_manager.set_teacher(super_teacher_model)

                super_cm = ClusteringMachine(
                    target_hardware="super_net", config=args, global_server=global_server,
                    clients_idx_arr=all_client_idx_arr, clients=clients,
                    start_round=args.start_round, last_round=args.last_round, path=args.path,task_id=task_id,
                    teacher_model=super_teacher_model,
                )
                
                super_cm.run()
                super_server = super_cm.get_server()
                
                if super_teacher_model is not None:
                    super_server.run_manager.set_teacher(super_teacher_model)
                    
                # 训练完成后保存 EWC/正交状态，供下一任务继承
                if args.ewc_lambda > 0:
                    fisher, processed = super_server.run_manager.compute_importance(
                        max_samples=args.ewc_samples_per_task
                    )
                    if fisher is not None:
                        super_server.run_manager.consolidate_ewc(fisher, update_prev_params=True)
                        ewc_state = super_server.run_manager.export_ewc_state()
                        _save_state_safely(ewc_state, ewc_state_path, desc="Supernet")
                    else:
                        print(f"[Supernet] Fisher is None (processed={processed}), skip EWC save")
                        
                if args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
                    ortho_ref, processed = super_server.run_manager.compute_ortho_reference(
                        max_samples=args.ortho_samples_per_task
                    )
                    if ortho_ref is not None:
                        ortho_state = super_server.run_manager.export_ortho_state()
                        _save_state_safely(ortho_state, ortho_state_path, desc="Supernet")
                    else:
                        print(f"[Supernet] Ortho ref is None (processed={processed}), skip ortho save")
                print("完成super_net训练阶段")
                # 保存本阶段的 replay buffer，便于断点续跑/下一任务复用
                for idx, cli in enumerate(clients):
                    replay_buffers_across_tasks[idx] = getattr(cli.run_manager, "replay_buffer", None)
                _save_replay_buffers(args.path)
                
                # 二、固化子网阶段
                global_server.load_model()
                global_server.get_normal_net()
                print('获取固化网络成功')
            
            # 三、子网重训阶段
            # 重训阶段可选覆盖
            _apply_phase_overrides("retrain")
            current_task_path = args.path  # 记录本任务根目录，便于重训结束后保存 replay
            retrain_path = args.path + '/learned_net'
            # 如果跳过 search，但 learned_net 不存在，则尝试用当前 global_server 固化出子网
            if args.skip_search and not os.path.exists(os.path.join(retrain_path, "net.config")):
                os.makedirs(retrain_path, exist_ok=True)
                global_server.load_model()
                global_server.get_normal_net()
            args.path = retrain_path
            os.makedirs(args.path, exist_ok=True)
            
            args.search = False
            retrain_start_round = args.start_round if args.retrain_start_round is None else args.retrain_start_round
            retrain_last_round = args.last_round if args.retrain_last_round is None else args.retrain_last_round
            
            args.client_id = 0
            global_run_config = CifarRunConfig(
                **args.__dict__,
                is_client = False
            )
            _attach_replay_cfg(global_run_config, args)
            
            # 加载子网结构与初始权重
            net_config_path = '%s/net.config' % args.path
            net = None
            if os.path.isfile(net_config_path):
                from models import get_net_by_name

                net_config = json.load(open(net_config_path, 'r'))
                net = get_net_by_name(net_config['name']).build_from_config(net_config)
                # 载入超网固化下来的初始权重（由 get_normal_net 保存到 learned_net/init）
                init_weight_path = os.path.join(args.path, "init")
                if os.path.isfile(init_weight_path):
                    try:
                        ckpt = torch.load(init_weight_path, map_location="cpu")
                        state_dict = ckpt.get("state_dict", ckpt)
                        missing, unexpected = net.load_state_dict(state_dict, strict=False)
                        sig = model_signature(net)
                        print(f"[Retrain] Loaded init weights from {init_weight_path}, "
                              f"missing: {len(missing)}, unexpected: {len(unexpected)}, "
                              f"sig_global_norm={sig.get('global', {}).get('norm'):.4f}")
                    except Exception as e:
                        print(f"[Retrain] Failed to load init weights from {init_weight_path}: {e}")
                else:
                    print(f"[Retrain] init weight {init_weight_path} not found, start from random init")
            else:
                print('net_config_path is not file!')

            # 记录重训开始前的子网签名
            with open(os.path.join(args.path, f"inherit_check_task{task_id}.log"), "a") as fout:
                fout.write(json.dumps({
                    "stage": "retrain_init_net",
                    "task_id": task_id,
                    "signature": model_signature(net),
                }) + "\n")

            # teacher：上一任务子网，用于 KD / kd_ortho
            teacher_model = None
            need_teacher = args.cl_kd_logit_lambda > 0 or args.cl_ortho_method == "kd_ortho"
            if need_teacher and task_id > 1:
                prev_retrain_path = prev_task_path + "/learned_net"
                teacher_model = copy.deepcopy(net)
                loaded_teacher = load_prev_task(teacher_model, prev_retrain_path)
                if not loaded_teacher:
                    teacher_model = None
                    print(f"[Retrain] 未能从 {prev_retrain_path} 加载教师模型，KD/kd_ortho 将跳过")
            
            # 全局 run_manager：基于当前子网初始化，后续可能加载 resume
            global_run_manager = RunManager(
                args.path, copy.deepcopy(net), global_run_config, init_model=False, task_id=task_id
            )
            global_run_manager.save_config(print_info=True)
            if teacher_model is not None:
                global_run_manager.set_teacher(teacher_model)
            
            # resume 时加载 checkpoint，并用加载后的权重作为初始化
            base_retrain_state = copy.deepcopy(global_run_manager.net.module.state_dict())
            
            clients, \
            mobile_client_idx_arr, cpu_client_idx_arr, \
            gpu8_client_idx_arr, flops_client_idx_arr, \
            None_client_idx_arr, all_client_idx_arr = \
                [], [], [], [], [], [], []

            for idx in range(args.num_users):
                all_client_idx_arr.append(idx)
                idx_target_hardware = set_target_hardware(idx=idx)
                if idx_target_hardware == 'mobile':
                    mobile_client_idx_arr.append(idx)
                elif idx_target_hardware == 'cpu':
                    cpu_client_idx_arr.append(idx)
                elif idx_target_hardware == 'gpu8':
                    gpu8_client_idx_arr.append(idx)
                elif idx_target_hardware == 'flops':
                    flops_client_idx_arr.append(idx)
                elif idx_target_hardware == None:
                    None_client_idx_arr.append(idx)
                # 每个客户端使用全局权重的深拷贝进行本地训练，避免互相覆盖
                client_net = copy.deepcopy(global_run_manager.net.module)
                client_net.load_state_dict(base_retrain_state, strict=False)
                client = RunManager(
                    args.path,
                    client_net,
                    clients_run_config_arr[idx],
                    init_model=False,
                    task_id=task_id,
                    replay_buffer=replay_buffers_across_tasks[idx],
                )
                _attach_replay_cfg(client.run_config, args)
                # retrain 阶段需要关闭 search，以便 valid() 使用测试集
                client.run_config.search = False
                clients.append(client)
                if teacher_model is not None:
                    client.set_teacher(teacher_model)
                print("The {} user has {} training data and {} test data.".format(idx,
                    client.run_config.data_provider.trn_set_length,
                    client.run_config.data_provider.tst_set_length))

            # helper：切换 run_manager 的 task_id 并强制刷新数据加载器
            def _reset_run_manager_task(run_mgr: RunManager, new_task_id: int):
                run_mgr.task_id = new_task_id
                run_mgr.run_config.task_id = new_task_id
                run_mgr.run_config.search = False
                # 清空缓存，让数据按新的 task_id 重新划分
                run_mgr.run_config._data_provider = None
                run_mgr.run_config._train_iter = None
                run_mgr.run_config._valid_iter = None
                run_mgr.run_config._test_iter = None
            def _rebuild_replay_buffer(run_mgr: RunManager, tasks_to_fill: list):
                buf = getattr(run_mgr, "replay_buffer", None)
                if (
                    buf is None
                    or getattr(run_mgr, "replay_mode", "none") == "none"
                    or getattr(run_mgr, "replay_capacity", 0) <= 0
                    or not tasks_to_fill
                ):
                    return {}
                buf.clear()
                stats = {}
                prev_task = run_mgr.task_id
                quota = buf.capacity // len(tasks_to_fill) if buf.capacity > 0 else 0
                for t in tasks_to_fill:
                    _reset_run_manager_task(run_mgr, t)
                    added = 0
                    for images, labels in run_mgr.run_config.train_loader:
                        buf.add_batch(images.detach().cpu(), labels.detach().cpu(), t)
                        added += images.size(0)
                        if quota > 0 and added >= quota:
                            break
                        if buf.capacity > 0 and len(buf) >= buf.capacity:
                            break
                    stats[t] = added
                    if buf.capacity > 0 and len(buf) >= buf.capacity:
                        break
                _reset_run_manager_task(run_mgr, prev_task)
                hist = buf.task_hist() if hasattr(buf, "task_hist") else {}
                print(
                    f"[ReplayPrefill] tasks={tasks_to_fill} quota={quota} buffer_size={len(buf)} hist={hist}"
                )
                return stats
            def _build_stage_training_buffer(run_mgr: RunManager, task_to_fill: int):
                main_buf = getattr(run_mgr, "replay_buffer", None)
                storage = getattr(main_buf, "storage", []) if main_buf is not None else []
                task_entries = [e for e in storage if int(e.get("task", -1)) == int(task_to_fill)]
                if not task_entries:
                    print(
                        f"[StageBuffer] task={task_to_fill} 无可用样本，stage_buffer 将为空"
                    )
                    return None
                stage_buf = SimpleReplayBuffer(len(task_entries))
                stage_buf.storage = [
                    {
                        "x": entry["x"].clone(),
                        "y": entry["y"].clone(),
                        "task": int(entry.get("task", task_to_fill)),
                        "t": int(entry.get("t", 0)),
                    }
                    for entry in task_entries
                ]
                stage_buf._time = max(entry.get("t", 0) for entry in task_entries)
                hist = stage_buf.task_hist() if hasattr(stage_buf, "task_hist") else {}
                print(
                    f"[StageBuffer] task={task_to_fill} stage_size={len(stage_buf)} hist={hist}"
                )
                return stage_buf

            retrain_task_schedule = list(range(1, task_id + 1)) if args.retrain_sequence_from_task1 else [task_id]
            # 后续 task 的多轮重训不再清理 log，避免覆盖上一任务的记录
            args.skip_retrain_log_cleanup = False
            # 保存跨任务的 EWC / 正交统计
            ewc_state = None
            ortho_state = None

            def _broadcast_ewc_state(state):
                global_run_manager.load_ewc_state(state)
                for rm in clients:
                    rm.load_ewc_state(state)
            def _broadcast_ortho_state(state):
                global_run_manager.load_ortho_state(state)
                for rm in clients:
                    rm.load_ortho_state(state)
            def _set_teacher_all(teacher):
                global_run_manager.set_teacher(teacher)
                for rm in clients:
                    rm.set_teacher(teacher)

            # 加载上一任务的 EWC / ortho
            ewc_state_path = os.path.join(args.path, "ewc_state.pt")
            prev_ewc_state_path = os.path.join(prev_task_path + "/learned_net", "ewc_state.pt")
            ortho_state_path = os.path.join(args.path, "ortho_state.pt")
            prev_ortho_state_path = os.path.join(prev_task_path + "/learned_net", "ortho_state.pt")
            ewc_state = _load_state_with_fallback(ewc_state_path, prev_ewc_state_path, desc="Retrain")
            ortho_state = _load_state_with_fallback(ortho_state_path, prev_ortho_state_path, desc="Retrain")

            _broadcast_ewc_state(ewc_state)
            _broadcast_ortho_state(ortho_state)
            if teacher_model is not None:
                _set_teacher_all(teacher_model)
            # retrain_sequence_from_task1 关闭时，先遍历前序任务，为 replay buffer 注入旧样本
            if not args.retrain_sequence_from_task1 and task_id > 1:
                print(f"[ReplayPrefill] retrain_sequence_from_task1=OFF，预先填充任务 1~{task_id - 1} 的样本到 replay buffer")
                tasks_to_fill = list(range(1, task_id))
                _rebuild_replay_buffer(global_run_manager, tasks_to_fill)
                for client_rm in clients:
                    _rebuild_replay_buffer(client_rm, tasks_to_fill)
                _reset_run_manager_task(global_run_manager, task_id)
                for client_rm in clients:
                    _reset_run_manager_task(client_rm, task_id)
            elif args.retrain_sequence_from_task1 and task_id > 1:
                print(f"[ReplayPrefill] retrain_sequence_from_task1=ON，构建 1~{task_id - 1} 混合样本用于随机重放")
                tasks_to_fill = list(range(1, task_id))
                _rebuild_replay_buffer(global_run_manager, tasks_to_fill)
                for client_rm in clients:
                    _rebuild_replay_buffer(client_rm, tasks_to_fill)

            for stage_idx, retrain_task_id in enumerate(retrain_task_schedule):
                print(f"[Retrain] 开始顺序重训 task {retrain_task_id}/{task_id}")
                if stage_idx > 0:
                    args.skip_retrain_log_cleanup = True
                replay_only_stage = (
                    args.retrain_sequence_from_task1 and retrain_task_id < task_id
                )
                _reset_run_manager_task(global_run_manager, retrain_task_id)
                if args.retrain_sequence_from_task1:
                    global_run_manager.reset_forgetting_stats()
                if args.retrain_sequence_from_task1 and retrain_task_id < task_id:
                    global_run_manager.stage_training_buffer = _build_stage_training_buffer(
                        global_run_manager, retrain_task_id
                    )
                    global_run_manager.allow_mix_during_stage = True
                    for client_rm in clients:
                        client_rm.stage_training_buffer = _build_stage_training_buffer(client_rm, retrain_task_id)
                        client_rm.allow_mix_during_stage = True
                        _reset_run_manager_task(client_rm, retrain_task_id)
                        if args.retrain_sequence_from_task1:
                            client_rm.reset_forgetting_stats()
                    print(f"[ReplayPrefill] task{retrain_task_id} stage buffers ready")
                else:
                    for client_rm in clients:
                        _reset_run_manager_task(client_rm, retrain_task_id)
                        if args.retrain_sequence_from_task1:
                            client_rm.reset_forgetting_stats()
                    global_run_manager.stage_training_buffer = None
                    global_run_manager.allow_mix_during_stage = False
                    for client_rm in clients:
                        client_rm.stage_training_buffer = None
                        client_rm.allow_mix_during_stage = False
                global_run_manager.replay_only_training = replay_only_stage
                for client_rm in clients:
                    client_rm.replay_only_training = replay_only_stage
                # 在当前任务开始前同步上一任务的 EWC 状态
                _broadcast_ewc_state(ewc_state)
                _broadcast_ortho_state(ortho_state)

                all_clients = CommonwealthMachine(
                    target_hardware='supernet',
                    config=args,
                    global_run_manager=global_run_manager,
                    clients_idx_arr=all_client_idx_arr,
                    clients=clients,
                    start_round=retrain_start_round,
                    last_round=retrain_last_round,
                    path=args.path,
                )
                
                all_clients.run()
                # 使用本阶段训练完的全局模型继续下一任务
                global_run_manager = all_clients.get_server()
                # 保存当前阶段模型，供下个阶段作为 teacher
                teacher_snapshot_path = os.path.join(args.path, f"teacher_task{retrain_task_id}.pth")
                try:
                    torch.save({"state_dict": global_run_manager.net.module.state_dict()}, teacher_snapshot_path)
                    print(f"[Retrain] Saved teacher snapshot to {teacher_snapshot_path}")
                except Exception as e:
                    print(f"[Retrain] Failed to save teacher snapshot: {e}")
                # 若还有后续阶段，更新 teacher 到最新模型
                if stage_idx < len(retrain_task_schedule) - 1:
                    teacher_model = copy.deepcopy(global_run_manager.net.module)
                    _set_teacher_all(teacher_model)
                # 训练完成后估计 Fisher，并更新 EWC 状态供下一任务使用
                if args.ewc_lambda > 0:
                    fisher, processed = global_run_manager.compute_importance(max_samples=args.ewc_samples_per_task)
                    global_run_manager.consolidate_ewc(fisher)
                    ewc_state = global_run_manager.export_ewc_state()
                    _save_state_safely(ewc_state, ewc_state_path, desc="Retrain")
                # 训练完成后估计 ortho 参考
                if args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
                    ortho_ref, processed = global_run_manager.compute_ortho_reference(
                        max_samples=args.ortho_samples_per_task
                    )
                    if ortho_ref is not None:
                        ortho_state = global_run_manager.export_ortho_state()
                        _save_state_safely(ortho_state, ortho_state_path, desc="Retrain")
                print(f"[Retrain] task {retrain_task_id} 重训完成")
                global_run_manager.replay_only_training = False
                for client_rm in clients:
                    client_rm.replay_only_training = False
                global_run_manager.stage_training_buffer = None
                global_run_manager.allow_mix_during_stage = False
                for client_rm in clients:
                    client_rm.stage_training_buffer = None
                    client_rm.allow_mix_during_stage = False

            print('所有客户端重训完成')
            # 记录当前任务后每个客户端的 replay buffer，供下一任务复用
            for idx, rm in enumerate(clients):
                replay_buffers_across_tasks[idx] = rm.replay_buffer
            _save_replay_buffers(current_task_path)

        elif args.object_to_search == "baseline":
            print(line_info()); print("-----------------------------case baseline: fixed backbone--------------------------------")
            _apply_phase_overrides("retrain")
            args.search = False
            args.client_id = 0
            global_run_config = CifarRunConfig(**args.__dict__, is_client=False)
            _attach_replay_cfg(global_run_config, args)
            global_net = BaselineResNet(
                arch=args.baseline_arch,
                num_classes=global_run_config.data_provider.n_classes,
                pretrained=args.baseline_pretrained,
            )
            
            teacher_model = None
            loaded_prev = False
            # 加载上一任务的模型权重
            if task_id > 1:
                try:
                    loaded_prev = load_prev_task(global_net, prev_task_path)
                except Exception as e:
                    print(f"[Baseline] 尝试从 {prev_task_path} 加载上一任务权重失败: {e}")
                    loaded_prev = False

            if not loaded_prev:
                print(f"[Baseline] 未找到上一任务权重，task {task_id} 从随机初始化开始")
                global_net.init_model(args.model_init, args.init_div_groups)
            else:
                print(f"[Baseline] 已从 {prev_task_path} 继承权重，task {task_id} 在上一任务模型上继续训练")
                
                # 初始化教师模型
                need_teacher = args.cl_kd_logit_lambda > 0 or args.cl_ortho_method == "kd_ortho"
                if need_teacher:
                    teacher_model = BaselineResNet(
                        arch=args.baseline_arch,
                        num_classes=global_run_config.data_provider.n_classes,
                        pretrained=args.baseline_pretrained,
                    )
                    teacher_model.load_state_dict(copy.deepcopy(global_net.state_dict()), strict=False)
                    
                    print(f"[Baseline] 已从 {prev_task_path} 继承教师模型")
                
            base_fixed_state = copy.deepcopy(global_net.state_dict())
            global_run_manager = RunManager(
                args.path, global_net, global_run_config, init_model=False, task_id=task_id
            )
            global_run_manager.save_config(print_info=True)
            if teacher_model is not None:
                global_run_manager.set_teacher(teacher_model)

            clients, all_client_idx_arr = [], []
            for idx in range(args.num_users):
                args.client_id = idx
                client_run_config = CifarRunConfig(**args.__dict__, is_client=True)
                _attach_replay_cfg(client_run_config, args)
                client_run_config.search = False
                client_net = BaselineResNet(
                    arch=args.baseline_arch,
                    num_classes=client_run_config.data_provider.n_classes,
                    pretrained=args.baseline_pretrained,
                )
                client_net.load_state_dict(base_fixed_state, strict=False)
                client = RunManager(
                    args.path,
                    client_net,
                    client_run_config,
                    init_model=False,
                    task_id=task_id,
                    replay_buffer=replay_buffers_across_tasks[idx],
                )
                clients.append(client)
                all_client_idx_arr.append(idx)
                print("The {} user has {} training data and {} test data.".format(
                    idx,
                    client.run_config.data_provider.trn_set_length,
                    client.run_config.data_provider.tst_set_length,
                ))
                
                # 为每个客户端下发教师模型
                if teacher_model is not None:
                    for client in clients:
                        client.set_teacher(teacher_model)

            all_clients = CommonwealthMachine(
                target_hardware=args.baseline_arch,
                config=args,
                global_run_manager=global_run_manager,
                clients_idx_arr=all_client_idx_arr,
                clients=clients,
                start_round=args.retrain_start_round,
                last_round=args.retrain_last_round,
                path=args.path,
            )
            
            ewc_state_path = os.path.join(args.path, "ewc_state.pt")
            prev_ewc_state_path = os.path.join(prev_task_path, "ewc_state.pt")
            ewc_state = None
            
            # 加载上一任务的 EWC 状态
            if os.path.isfile(ewc_state_path):
                try:
                    ewc_state = torch.load(ewc_state_path, map_location="cpu")
                    print(f"[Baseline] Loaded EWC state from {ewc_state_path}")
                except Exception as e:
                    print(f"[Baseline] Failed to load EWC state: {e}")
            elif os.path.isfile(prev_ewc_state_path):
                try:
                    ewc_state = torch.load(prev_ewc_state_path, map_location="cpu")
                    print(f"[Baseline] Loaded EWC state from prev task {prev_ewc_state_path}")
                except Exception as e:
                    print(f"[Baseline] Failed to load prev-task EWC state: {e}")
                    
            def _broadcast_ewc_state(state):
                global_run_manager.load_ewc_state(state)
                for rm in clients:
                    rm.load_ewc_state(state)

            _broadcast_ewc_state(ewc_state)
            
            # 加载上一任务的正交参考状态
            ortho_state = None
            ortho_state_path = os.path.join(prev_task_path, "ortho_state.pt")  # 或当前 task 目录
            if os.path.isfile(ortho_state_path):
                ortho_state = torch.load(ortho_state_path, map_location="cpu")
                
            # 给 global & clients 广播
            global_run_manager.load_ortho_state(ortho_state)
            for client in clients:
                client.load_ortho_state(ortho_state)
            
            all_clients.run()
            # 记录当前任务的 replay buffer，供下一任务复用
            for idx, rm in enumerate(clients):
                replay_buffers_across_tasks[idx] = rm.replay_buffer
            _save_replay_buffers(args.path)
            
            # 保存 EWC 正则化模型
            if args.ewc_lambda > 0:
                print(f"[Baseline] 计算并整合 Fisher 信息，lambda={args.ewc_lambda}")
                fisher, processed = global_run_manager.compute_importance(max_samples=args.ewc_samples_per_task)
                if fisher is None:
                    print(f"[Baseline] Fisher/importance is None (processed={processed}), skip consolidate")
                else:
                    global_run_manager.consolidate_ewc(fisher, update_prev_params=True)
                    ewc_state = global_run_manager.export_ewc_state()
                    print(f"[Baseline] Importance_keys={len(fisher)}, importance_norm={sum(v.sum().item() for v in fisher.values()):.4f}, processed={processed}")
                    try:
                        torch.save(ewc_state, ewc_state_path)
                        print(f"[Baseline] Saved EWC state to {ewc_state_path}")
                    except Exception as e:
                        print(f"[Baseline] Failed to save EWC state: {e}")
                        
            # 保存正交模型
            if args.cl_ortho_method != "none" and args.ortho_samples_per_task > 0:
                ortho_ref, processed = global_run_manager.compute_ortho_reference(
                    max_samples=args.ortho_samples_per_task
                )
                if ortho_ref is not None and isinstance(ortho_ref, dict):
                    # 统计范数（若存在 global 键则用 global，否则取所有梯度范数之和）
                    if "global" in ortho_ref:
                        ref_norm = ortho_ref["global"].norm()
                    else:
                        ref_norm = sum(v.norm() for v in ortho_ref.values())
                    print(f"[Baseline] Ortho ref norm={ref_norm:.4f}, processed={processed}")
                    ortho_state = global_run_manager.export_ortho_state()
                    ortho_state_path = os.path.join(args.path, "ortho_state.pt")
                    try:
                        torch.save(ortho_state, ortho_state_path)
                        print(f"[Baseline] Saved ortho state to {ortho_state_path}")
                    except Exception as e:
                        print(f"[Baseline] Failed to save ortho state: {e}")
                else:
                    print(f"[Baseline] Ortho ref is None (processed={processed}), skip saving ortho_state")


        elif args.object_to_search == "cpu":
            _run_pruning_case("cpu", cpu_client_idx_arr, "-----------------------------case2: pruning for cpu--------------------------------")

        elif args.object_to_search == "gpu8":
            _run_pruning_case("gpu8", gpu8_client_idx_arr, "---------------------------------case3: pruning for gpu8------------------------------------------")

        elif args.object_to_search == "flops":
            _run_pruning_case("flops", flops_client_idx_arr, "------------------------case4: pruning for flops--------------------------------------")

        elif args.object_to_search is None:
            _run_pruning_case(None, none_client_idx_arr, "--------------------------------------case5: pruning for None------------------------------------------")

        print(f"任务 {task_id}/{args.num_tasks} 完成！")

if __name__ == "__main__":
    main()
