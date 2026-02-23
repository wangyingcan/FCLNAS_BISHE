"""
读取成本画像CSV，做线性回归并输出可视化。
需要 cost_profile_subnets.csv 与 cost_profile_batch.csv。
"""
import argparse
import os
import json
import math
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 优先使用常见中文/英文字体（适合论文图示）
rcParams['font.sans-serif'] = ['SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

# 确保项目根目录在 sys.path，便于作为脚本直接运行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return slope, intercept, r2


def linear_fit_2d(x1: np.ndarray, x2: np.ndarray, y: np.ndarray):
    """y = a*x1 + b*x2 + c"""
    A = np.vstack([x1, x2, np.ones_like(x1)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b, c = coef
    y_pred = a * x1 + b * x2 + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return a, b, c, r2, y_pred


def scatter_2d_color(x, y, z, xlabel, ylabel, zlabel, save_path, eq_text=None, cmap='viridis'):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=z, cmap=cmap, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    cb = plt.colorbar(sc)
    cb.set_label(zlabel)
    if eq_text:
        plt.title(eq_text)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def scatter_2d_color_clean(x, y, z, save_path, cmap='viridis'):
    """无标题、无中文文案的 heat 图，保留坐标刻度。"""
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(x, y, c=z, cmap=cmap, alpha=0.85)
    cb = plt.colorbar(sc)
    cb.set_label('')
    # 保留刻度，去掉标签
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def format_linear_eq(a, b, c, target='T'):
    def term(coef, symbol):
        sign = '+' if coef >= 0 else '-'
        return f" {sign} {abs(coef):.4f}{symbol}"
    return f"${target} ={term(a, 'F')}{term(b, 'P')}{term(c, '')}$"


def plot_scatter_fit(x, y, xlabel, ylabel, save_path, title=None,
                     point_color='tab:blue', line_color='tab:red'):
    slope, intercept, r2 = linear_fit(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, label='samples', color=point_color)
    plt.plot(x_line, y_line, line_color, label=f'fit: y={slope:.4f}x+{intercept:.2f}\nR^2={r2:.3f}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return slope, intercept, r2


def plot_curve(x, y, xlabel, ylabel, save_path, title=None):
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    plt.figure(figsize=(6, 4))
    plt.plot(x_sorted, y_sorted, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def load_csv(path):
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='cost_profile/outputs')
    parser.add_argument('--subnet_csv', type=str, default=None, help='默认 input_dir/cost_profile_subnets.csv')
    parser.add_argument('--batch_csv', type=str, default=None, help='默认 input_dir/cost_profile_batch.csv')
    parser.add_argument('--output_dir', type=str, default=None, help='图像输出目录，默认 input_dir/plots')
    args = parser.parse_args()

    subnet_csv = args.subnet_csv or os.path.join(args.input_dir, 'cost_profile_subnets.csv')
    batch_csv = args.batch_csv or os.path.join(args.input_dir, 'cost_profile_batch.csv')
    out_dir = args.output_dir or os.path.join(args.input_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    # 实验1
    sub_rows = load_csv(subnet_csv)
    f_vals = np.array([float(r['flops']) / 1e6 for r in sub_rows])  # MFLOPs per batch
    t_vals = np.array([float(r['t_step_ms']) for r in sub_rows])
    p_vals = np.array([float(r['params']) / 1e6 for r in sub_rows])  # M params
    m_vals = np.array([float(r['m_step_mb']) for r in sub_rows])

    slope_t, inter_t, r2_t = plot_scatter_fit(
        f_vals, t_vals,
        xlabel='FLOPs per batch (M)',
        ylabel='Avg step time (ms)',
        save_path=os.path.join(out_dir, 'f_vs_t.png'),
        title=None,
        point_color='tab:blue',
        line_color='tab:purple'
    )
    slope_m, inter_m, r2_m = plot_scatter_fit(
        p_vals, m_vals,
        xlabel='Params (M)',
        ylabel='Peak mem (MB)',
        save_path=os.path.join(out_dir, 'p_vs_m.png'),
        title=None,
        point_color='tab:orange',
        line_color='tab:green'
    )

    # 多元线性回归 & 可视化：FLOPs & Params -> T / M
    a_t, b_t, c_t, r2_t2, t_pred = linear_fit_2d(f_vals, p_vals, t_vals)
    a_m, b_m, c_m, r2_m2, m_pred = linear_fit_2d(f_vals, p_vals, m_vals)

    scatter_2d_color(
        f_vals, p_vals, t_vals,
        xlabel='模型浮点数运算量 F\n(M FLOPs)', ylabel='模型参数量 P\n(M params)', zlabel='时延 T\n(ms)',
        save_path=os.path.join(out_dir, 'f_p_t_heat.png'),
        eq_text=f'{format_linear_eq(a_t, b_t, c_t, target="T")}\\n$R^2={r2_t2:.3f}$',
        cmap='viridis'
    )
    scatter_2d_color(
        f_vals, p_vals, m_vals,
        xlabel='模型浮点数运算量 F\n(M FLOPs)', ylabel='模型参数量 P\n(M params)', zlabel='显存峰值 M_max\n(MB)',
        save_path=os.path.join(out_dir, 'f_p_m_heat.png'),
        eq_text=f'{format_linear_eq(a_m, b_m, c_m, target="M_{max}")}\\n$R^2={r2_m2:.3f}$',
        cmap='magma'
    )
    # 纯净版（无文案）heat图
    scatter_2d_color_clean(
        f_vals, p_vals, t_vals,
        save_path=os.path.join(out_dir, 'f_p_t_heat_clean.png'),
        cmap='viridis'
    )
    scatter_2d_color_clean(
        f_vals, p_vals, m_vals,
        save_path=os.path.join(out_dir, 'f_p_m_heat_clean.png'),
        cmap='magma'
    )

    # 实验2（如果有批次扫描数据才画图）
    if os.path.exists(batch_csv):
        batch_rows = load_csv(batch_csv)
        b_vals = np.array([float(r['batch_size']) for r in batch_rows])
        t_batch = np.array([float(r['t_step_ms']) for r in batch_rows])
        m_batch = np.array([float(r['m_step_mb']) for r in batch_rows])

        plot_curve(b_vals, t_batch, xlabel='Batch size', ylabel='Avg step time (ms)',
                   save_path=os.path.join(out_dir, 'b_vs_t.png'))
        plot_curve(b_vals, m_batch, xlabel='Batch size', ylabel='Peak mem (MB)',
                   save_path=os.path.join(out_dir, 'b_vs_m.png'))
    else:
        print(f'未找到批次扫描文件 {batch_csv}，跳过 B-vs-cost 图。')

    summary = {
        'f_vs_t': {'slope': slope_t, 'intercept': inter_t, 'r2': r2_t},
        'p_vs_m': {'slope': slope_m, 'intercept': inter_m, 'r2': r2_m},
        'fp_vs_t': {'a_f': a_t, 'b_p': b_t, 'c': c_t, 'r2': r2_t2},
        'fp_vs_m': {'a_f': a_m, 'b_p': b_m, 'c': c_m, 'r2': r2_m2},
    }
    summary_path = os.path.join(out_dir, 'regression_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('回归结果:', summary)
    print(f'图像保存在 {out_dir}')


if __name__ == '__main__':
    main()
