"""
全流程采集成本数据：
- 实验1：固定 batch=8，224x224 输入，采样 N 子网(+轻/重极端)，记录 F(G), P(M), T_step(ms), M_max(MB)
- 实验2：从实验1中挑代表子网，扫描多种 batch，记录同上
输出：
- cost_profile/outputs/cost_profile_subnets.csv
- cost_profile/outputs/cost_profile_batch.csv
- 子网配置保存到 cost_profile/outputs/subnets/expX_*.json
"""
import argparse
import csv
import os
import random
from statistics import median
import sys

import torch

# ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cost_profile.profiling import build_supernet, sample_subnet, measure_step_cost, profile_flops_params


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    import json
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def experiment1(args):
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    super_net = build_supernet(n_classes=100, width_mult=args.width_mult)
    super_net.to(device)

    out_csv = os.path.join(args.output_dir, 'cost_profile_subnets.csv')
    subnet_dir = os.path.join(args.output_dir, 'subnets')
    ensure_dir(subnet_dir)

    rows = []

    def profile_one(subnet, sid):
        subnet.to(device)
        F_g, P_m = profile_flops_params(subnet, input_size=(3, 224, 224))
        res = measure_step_cost(
            subnet,
            batch_size=args.exp1_batch,
            steps=args.measure_steps,
            warmup=args.warmup_steps,
            input_size=(3, 224, 224),
            cudnn_benchmark=False,
            cudnn_deterministic=True,
        )
        config_path = os.path.join(subnet_dir, f'exp1_{sid}.json')
        save_json(subnet.config, config_path)
        rows.append({
            'model_id': sid,
            'F_G': F_g,
            'P_M': P_m,
            'batch_size': args.exp1_batch,
            'T_step_ms': res.avg_step_ms,
            'M_max_MB': res.max_mem_mb,
            'config_path': config_path,
        })
        print(f"[Exp1 {sid}] F={F_g:.2f}G P={P_m:.2f}M T={res.avg_step_ms:.2f}ms M={res.max_mem_mb:.1f}MB")
        torch.cuda.empty_cache()

    # light / heavy
    light = sample_subnet(super_net, rng=lambda m: 0)
    heavy = sample_subnet(super_net, rng=lambda m: m.n_choices - 1)
    profile_one(light, 'light')
    profile_one(heavy, 'heavy')

    # random subnets
    for i in range(args.num_subnets):
        subnet = sample_subnet(super_net, rng)
        profile_one(subnet, i)

    # save csv
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"[Exp1] saved {len(rows)} rows to {out_csv}")
    return rows


def pick_representative(rows, k=6):
    # 按 FLOPs 排序，均匀抽取 k 个
    rows_sorted = sorted(rows, key=lambda r: r['F_G'])
    if k >= len(rows_sorted):
        return rows_sorted
    idxs = [int(i * (len(rows_sorted)-1) / (k-1)) for i in range(k)]
    reps = [rows_sorted[i] for i in idxs]
    return reps


def experiment2(args, reps):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_csv = os.path.join(args.output_dir, 'cost_profile_batch.csv')
    rows = []
    for r in reps:
        cfg = r['config_path']
        import json
        from models.normal_nets.proxyless_nets import ProxylessNASNets
        subnet = ProxylessNASNets.build_from_config(json.load(open(cfg, 'r'))).to(device)
        label = r['model_id']
        for b in args.exp2_batches:
            try:
                res = measure_step_cost(
                    subnet,
                    batch_size=b,
                    steps=args.measure_steps,
                    warmup=args.warmup_steps,
                    input_size=(3, 224, 224),
                    cudnn_benchmark=False,
                    cudnn_deterministic=True,
                )
                rows.append({
                    'model_id': label,
                    'F_G': r['F_G'],
                    'P_M': r['P_M'],
                    'batch_size': b,
                    'T_step_ms': res.avg_step_ms,
                    'M_max_MB': res.max_mem_mb,
                    'config_path': cfg,
                })
                print(f"[Exp2 {label}] B={b} T={res.avg_step_ms:.2f}ms M={res.max_mem_mb:.1f}MB")
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"[Exp2 {label}] B={b} OOM, skip")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            torch.cuda.empty_cache()
    if rows:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
        print(f"[Exp2] saved {len(rows)} rows to {out_csv}")
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--output_dir', type=str, default='cost_profile/outputs')
    p.add_argument('--num_subnets', type=int, default=50)
    p.add_argument('--width_mult', type=float, default=1.0)
    p.add_argument('--exp1_batch', type=int, default=8)
    p.add_argument('--exp2_batches', type=str, default='4,8,16,32')
    p.add_argument('--warmup_steps', type=int, default=30)
    p.add_argument('--measure_steps', type=int, default=150)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    args.exp2_batches = [int(x) for x in args.exp2_batches.split(',') if x]
    ensure_dir(args.output_dir)
    rows1 = experiment1(args)
    reps = pick_representative(rows1, k=6)
    experiment2(args, reps)


if __name__ == '__main__':
    main()
