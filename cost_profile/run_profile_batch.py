"""
Experiment 2: 固定子网，改变 batch size，测量 (B, T_step, M_step)。
需要提供子网 config JSON（可用实验1的 median_subnet_info.json 中的 config_path）。
"""
import argparse
import csv
import json
import os
import sys
from typing import List

import torch

# 确保项目根目录在 sys.path，便于作为脚本直接运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.normal_nets.proxyless_nets import ProxylessNASNets
from cost_profile.profiling import measure_step_cost


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subnet_configs', type=str, required=True,
                        help='多个子网config路径，逗号分隔（例如 light.json,mid.json,heavy.json）')
    parser.add_argument('--labels', type=str, default=None,
                        help='与子网对应的标签，逗号分隔（可选，默认用文件名）')
    parser.add_argument('--output_dir', type=str, default='cost_profile/outputs')
    parser.add_argument('--batch_list', type=str, default='4,8,16,32,64,128')
    parser.add_argument('--warmup_steps', type=int, default=30)
    parser.add_argument('--measure_steps', type=int, default=150)
    parser.add_argument('--input_size', type=str, default='3,32,32', help='通道,高,宽')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def load_subnet(config_path: str) -> ProxylessNASNets:
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    net = ProxylessNASNets.build_from_config(cfg)
    return net


def main():
    args = parse_args()
    batch_values = [int(x) for x in args.batch_list.split(',') if x]
    input_size = tuple(int(x) for x in args.input_size.split(','))
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'cost_profile_batch.csv')

    config_list: List[str] = [p for p in args.subnet_configs.split(',') if p]
    if args.labels:
        label_list = args.labels.split(',')
        assert len(label_list) == len(config_list), "labels 数量需与子网数量一致"
    else:
        label_list = [os.path.splitext(os.path.basename(p))[0] for p in config_list]

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    rows = []
    for cfg_path, label in zip(config_list, label_list):
        net = load_subnet(cfg_path).to(device)
        for b in batch_values:
            try:
                result = measure_step_cost(
                    net,
                    batch_size=b,
                    steps=args.measure_steps,
                    warmup=args.warmup_steps,
                    input_size=input_size,
                    cudnn_benchmark=False,
                    cudnn_deterministic=True,
                )
                rows.append({
                    'label': label,
                    'batch_size': b,
                    'flops': result.flops,
                    'params': result.params,
                    't_step_ms': result.avg_step_ms,
                    'm_step_mb': result.max_mem_mb,
                    'config_path': cfg_path,
                })
                print(f"[{label}] B={b}: T_step={result.avg_step_ms:.3f}ms, M_step={result.max_mem_mb:.1f}MB")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"[{label}] B={b}: OOM，跳过")
                    torch.cuda.empty_cache()
                    continue
                raise
            torch.cuda.empty_cache()

    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"保存CSV到 {csv_path}")
    else:
        print('没有成功的 batch 配置被记录。')


if __name__ == '__main__':
    main()
