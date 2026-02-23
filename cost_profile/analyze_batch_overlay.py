import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# 字体设置与主分析脚本一致
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_csv(path):
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def overlay_plot(rows, y_key, save_path, ylabel_unit):
    """三类子网同图，不做回归，去掉轴文案但保留刻度。"""
    plt.figure(figsize=(6, 4))
    labels = sorted(set(r['label'] for r in rows))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    xticks = [4, 8, 16, 32, 64, 128]
    for lab, col in zip(labels, colors):
        sub = [r for r in rows if r['label'] == lab]
        b = np.array([float(r['batch_size']) for r in sub])
        y = np.array([float(r[y_key]) for r in sub])
        order = np.argsort(b)
        plt.plot(b[order], y[order], marker='o', color=col, label=lab)
    plt.xticks(xticks)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def overlay_per_sample(rows, y_key, save_path):
    """三类子网的每样本开销曲线（y/batch），无轴文案，刻度保留。"""
    plt.figure(figsize=(6, 4))
    labels = sorted(set(r['label'] for r in rows))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    xticks = [4, 8, 16, 32, 64, 128]
    for lab, col in zip(labels, colors):
        sub = [r for r in rows if r['label'] == lab]
        b = np.array([float(r['batch_size']) for r in sub])
        y = np.array([float(r[y_key]) for r in sub]) / b
        order = np.argsort(b)
        plt.plot(b[order], y[order], marker='o', color=col, label=lab)
    plt.xticks(xticks)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def regression_plot(rows, y_key, save_path, ylabel_unit):
    """仅中等子网做回归，轴无文案，保留刻度"""
    sub = [r for r in rows if r['label'] == 'medium']
    b = np.array([float(r['batch_size']) for r in sub])
    y = np.array([float(r[y_key]) for r in sub])
    xticks = [4, 8, 16, 32, 64, 128]
    A = np.vstack([b, np.ones_like(b)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    x_line = np.linspace(b.min(), b.max(), 100)
    y_line = slope * x_line + intercept
    plt.figure(figsize=(6, 4))
    order = np.argsort(b)
    plt.plot(b[order], y[order], 'o', color='tab:blue', label='medium')
    plt.plot(x_line, y_line, color='tab:red', label=f'y={slope:.4f}x+{intercept:.2f}')
    plt.xticks(xticks)
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_csv', type=str, default='cost_profile/outputs/cost_profile_batch.csv')
    parser.add_argument('--output_dir', type=str, default='cost_profile/outputs/plots')
    args = parser.parse_args()

    rows = load_csv(args.batch_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    overlay_plot(rows, 't_step_ms', os.path.join(args.output_dir, 'batch_overlay_t.png'), 'ms')
    overlay_plot(rows, 'm_step_mb', os.path.join(args.output_dir, 'batch_overlay_m.png'), 'MB')
    overlay_per_sample(rows, 't_step_ms', os.path.join(args.output_dir, 'batch_overlay_t_per_sample.png'))
    overlay_per_sample(rows, 'm_step_mb', os.path.join(args.output_dir, 'batch_overlay_m_per_sample.png'))
    regression_plot(rows, 't_step_ms', os.path.join(args.output_dir, 'batch_medium_t.png'), 'ms')
    regression_plot(rows, 'm_step_mb', os.path.join(args.output_dir, 'batch_medium_m.png'), 'MB')


if __name__ == '__main__':
    main()
