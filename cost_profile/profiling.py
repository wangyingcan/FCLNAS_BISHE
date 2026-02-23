import json
import math
import os
import random
import time
import copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from torch import nn

from models.super_nets.super_proxyless import SuperProxylessNASNets
from models.normal_nets.proxyless_nets import ProxylessNASNets

# -----------------------------
# Subnet sampling helpers
# -----------------------------


def build_supernet(n_classes: int = 100,
                   width_mult: float = 1.0,
                   width_stages: List[int] = None,
                   n_cell_stages: List[int] = None,
                   stride_stages: List[int] = None,
                   conv_candidates: List[str] = None,
                   inference_device: str = 'gpu') -> SuperProxylessNASNets:
    """Create a ProxylessNAS supernet with sensible defaults for CIFAR-sized inputs."""
    width_stages = width_stages or [24, 40, 80, 96, 192, 320]
    n_cell_stages = n_cell_stages or [2, 3, 4, 3, 4, 3]
    stride_stages = stride_stages or [1, 1, 2, 1, 2, 1]
    conv_candidates = conv_candidates or [
        '3x3_MBConv2', '3x3_MBConv3', '3x3_MBConv4', '3x3_MBConv5', '3x3_MBConv6', '5x5_MBConv3'
    ]
    return SuperProxylessNASNets(
        width_stages=width_stages,
        n_cell_stages=n_cell_stages,
        stride_stages=stride_stages,
        conv_candidates=conv_candidates,
        n_classes=n_classes,
        width_mult=width_mult,
        inference_device=inference_device,
    )


def sample_subnet(super_net: SuperProxylessNASNets, rng) -> ProxylessNASNets:
    """Choose one op per MixedEdge then export a normal Proxyless subnet.

    rng 可以是 random.Random 实例（使用 randrange），或可调用对象 lambda m: idx 用于确定性极端子网。
    """
    net = copy.deepcopy(super_net)
    for m in net.redundant_modules:
        if callable(rng):
            idx = rng(m)
        else:
            idx = rng.randrange(m.n_choices)
        # force this op to be the chosen one by setting arch alpha
        with torch.no_grad():
            m.AP_path_alpha.data.zero_()
            m.AP_path_alpha.data[idx] = 10.0  # dominant prob
        m.active_index = [idx]
        m.inactive_index = [i for i in range(m.n_choices) if i != idx]
    # convert to normal net (no MixedEdge)
    normal_net = net.convert_to_normal_net()
    return normal_net


# -----------------------------
# Profiling utilities
# -----------------------------


@dataclass
class ProfileResult:
    flops: float
    params: int
    avg_step_ms: float
    max_mem_mb: float


@torch.inference_mode()
def _compute_flops_params(model: ProxylessNASNets, batch_size: int, input_size=(3, 32, 32)) -> Tuple[float, int]:
    device = next(model.parameters()).device
    dummy = torch.randn(1, *input_size, device=device)
    flops_per_sample, _ = model.get_flops(dummy)
    flops = float(flops_per_sample) * batch_size  # total FLOPs per batch
    params = sum(p.numel() for p in model.parameters())
    return flops, params


def measure_step_cost(model: ProxylessNASNets,
                      batch_size: int,
                      steps: int = 50,
                      warmup: int = 10,
                      input_size=(3, 32, 32),
                      cudnn_benchmark: bool = True,
                      cudnn_deterministic: bool = False) -> ProfileResult:
    device = next(model.parameters()).device
    model.train()
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic
    num_classes = model.classifier.out_features
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    def run_one_step():
        inputs = torch.randn(batch_size, *input_size, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss

    # warmup
    for _ in range(warmup):
        run_one_step()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(steps):
        run_one_step()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000.0 / steps
    if device.type == 'cuda':
        max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        max_mem_mb = 0.0

    flops, params = _compute_flops_params(model, batch_size, input_size)
    return ProfileResult(flops=flops, params=params, avg_step_ms=avg_ms, max_mem_mb=max_mem_mb)


# -----------------------------
# JSON / CSV helpers
# -----------------------------

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
