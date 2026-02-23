"""
Experiment 1: 固定 batch & 硬件，随机采样多组 Proxyless 子网，记录 (F, P, B, T_step, M_step)。
输出：
- CSV: cost_profile_subnets.csv
- 每个子网的 config JSON：subnets/subnet_<idx>.json
"""
import argparse
import csv
import os
import random
from statistics import median
import sys

import torch

# 确保项目根目录在 sys.path，便于作为脚本直接运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cost_profile.profiling import build_supernet, sample_subnet, measure_step_cost, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='cost_profile/outputs', help='日志和子网配置的输出目录')
    parser.add_argument('--num_subnets', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_size', type=str, default='3,32,32', help='通道,高,宽。例如 3,224,224')
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--measure_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--width_mult', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--add_extreme_subnets', action='store_true',
                        help='在随机采样之外，附加最轻/最重两个极端子网用于拉开跨度')
    parser.add_argument('--cudnn_deterministic', action='store_true', help='固定 cuDNN 算法，降低时延抖动')
    parser.add_argument('--no_cudnn_benchmark', action='store_true', help='关闭 cuDNN benchmark，配合 deterministic 使用')
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'cost_profile_subnets.csv')
    subnet_dir = os.path.join(args.output_dir, 'subnets')
    os.makedirs(subnet_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    super_net = build_supernet(n_classes=100, width_mult=args.width_mult)
    super_net.to(device)
    c, h, w = [int(x) for x in args.input_size.split(',')]
    input_size = (c, h, w)

    rows = []
    def profile_and_record(subnet, sid):
        subnet.to(device)
        try:
            result = measure_step_cost(
                subnet, batch_size=args.batch_size, steps=args.measure_steps,
                warmup=args.warmup_steps, input_size=input_size,
                cudnn_benchmark=not args.no_cudnn_benchmark,
                cudnn_deterministic=args.cudnn_deterministic
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"[{sid}] OOM，跳过（batch={args.batch_size}, input={input_size}）")
                torch.cuda.empty_cache()
                return
            else:
                raise
        config = subnet.config
        json_path = os.path.join(subnet_dir, f'subnet_{sid}.json')
        save_json(config, json_path)
        rows.append({
            'subnet_id': sid,
            'batch_size': args.batch_size,
            'flops': result.flops,
            'params': result.params,
            't_step_ms': result.avg_step_ms,
            'm_step_mb': result.max_mem_mb,
            'config_path': json_path,
        })
        print(f"[{sid}] FLOPs={result.flops/1e6:.1f}M, Params={result.params/1e6:.2f}M, "
              f"T_step={result.avg_step_ms:.3f}ms, M_step={result.max_mem_mb:.1f}MB")
        torch.cuda.empty_cache()

    # 可选：先加入最轻/最重子网作为锚点
    if args.add_extreme_subnets:
        # 最轻：每个 MixedEdge 选最小索引算子
        light = sample_subnet(super_net, rng=lambda m: 0)
        profile_and_record(light, 'light')
        # 最重：每个 MixedEdge 选最大索引算子
        heavy = sample_subnet(super_net, rng=lambda m: m.n_choices - 1)
        profile_and_record(heavy, 'heavy')

    # 随机子网
    for idx in range(args.num_subnets):
        subnet = sample_subnet(super_net, rng)
        profile_and_record(subnet, idx)

    # write csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # pick median subnet id for experiment 2 reference
    rows_sorted = sorted(rows, key=lambda r: r['flops'])
    median_row = rows_sorted[len(rows_sorted) // 2]
    median_path = os.path.join(args.output_dir, 'median_subnet_info.json')
    save_json(median_row, median_path)
    print(f"保存CSV到 {csv_path}; 中值子网信息写入 {median_path}")


if __name__ == '__main__':
    main()
