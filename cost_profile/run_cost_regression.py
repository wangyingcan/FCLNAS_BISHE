"""
读取实验1/2数据做多元线性回归，输出系数、R2、预测误差图。
模型：
T_step = aT*F + bT*P + cT*(1/B) + dT
M_max  = aM*F + bM*P + cM*B      + dM
输出：
- regression_cost.json
- 若干图：pred_vs_true、error_hist、F/P/T散点等（含带标签与无标签）
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import rcParams
from sklearn.model_selection import train_test_split

# 字体设置
rcParams['font.sans-serif'] = ['SimSun', 'Noto Sans CJK SC', 'Microsoft YaHei', 'PingFang SC', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['font.family'] = 'sans-serif'
rcParams['axes.unicode_minus'] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_csv(path):
    with open(path, 'r') as f:
        return list(csv.DictReader(f))


def build_features(rows):
    F = np.array([float(r['F_G']) for r in rows])
    P = np.array([float(r['P_M']) for r in rows])
    B = np.array([float(r['batch_size']) for r in rows])
    T = np.array([float(r['T_step_ms']) for r in rows])
    M = np.array([float(r['M_max_MB']) for r in rows])
    X_T = np.stack([F, P, 1.0 / B, np.ones_like(F)], axis=1)
    X_M = np.stack([F, P, B, np.ones_like(F)], axis=1)
    return X_T, X_M, T, M


def ols_fit(X, y):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    mse = np.mean((y - y_pred) ** 2)
    return coef, y_pred, r2, mse


def plot_pred_true(y_true, y_pred, save_path, title, clean=False):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--')
    if not clean:
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.title(title)
    else:
        plt.xlabel('')
        plt.ylabel('')
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def plot_hist(rel_err, save_path, title, clean=False):
    plt.figure(figsize=(5, 4))
    plt.hist(rel_err, bins=30, alpha=0.8, color='tab:blue', edgecolor='white')
    if not clean:
        plt.xlabel('Relative Error')
        plt.ylabel('Count')
        plt.title(title)
    else:
        plt.xlabel('')
        plt.ylabel('')
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def plot_scatter_color(x, y, c, xlabel, ylabel, clabel, save_path, clean=False, cmap='viridis'):
    plt.figure(figsize=(6, 4))
    sc = plt.scatter(x, y, c=c, cmap=cmap, alpha=0.75)
    cb = plt.colorbar(sc)
    if not clean:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        cb.set_label(clabel)
    else:
        plt.xlabel('')
        plt.ylabel('')
        cb.set_label('')
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def plot_3d_scatter(x, y, z, c, xlabel, ylabel, zlabel, clabel, save_path, clean=False, cmap='viridis'):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(x, y, z, c=c, cmap=cmap, alpha=0.75)
    if not clean:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        cb = fig.colorbar(p, ax=ax, shrink=0.6)
        cb.set_label(clabel)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subnet_csv', type=str, default='cost_profile/outputs/cost_profile_subnets.csv')
    ap.add_argument('--batch_csv', type=str, default='cost_profile/outputs/cost_profile_batch.csv')
    ap.add_argument('--output_dir', type=str, default='cost_profile/outputs/plots')
    ap.add_argument('--test_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_csv(args.subnet_csv) + load_csv(args.batch_csv)

    X_T, X_M, T, M = build_features(rows)

    # train/test split
    idx = np.arange(len(T))
    train_idx, test_idx = train_test_split(idx, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
    def split(X, y):
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    X_Ttr, X_Tte, Ttr, Tte = split(X_T, T)
    X_Mtr, X_Mte, Mtr, Mte = split(X_M, M)

    coef_T, pred_Ttr, r2_Ttr, mse_Ttr = ols_fit(X_Ttr, Ttr)
    _, pred_Tte, r2_Tte, mse_Tte = ols_fit(X_Tte, Tte)
    coef_M, pred_Mtr, r2_Mtr, mse_Mtr = ols_fit(X_Mtr, Mtr)
    _, pred_Mte, r2_Mte, mse_Mte = ols_fit(X_Mte, Mte)

    summary = {
        'T_step': {
            'coef': {'a_T': coef_T[0], 'b_T': coef_T[1], 'c_T': coef_T[2], 'd_T': coef_T[3]},
            'r2_train': r2_Ttr, 'mse_train': mse_Ttr,
            'r2_test': r2_Tte, 'mse_test': mse_Tte,
        },
        'M_max': {
            'coef': {'a_M': coef_M[0], 'b_M': coef_M[1], 'c_M': coef_M[2], 'd_M': coef_M[3]},
            'r2_train': r2_Mtr, 'mse_train': mse_Mtr,
            'r2_test': r2_Mte, 'mse_test': mse_Mte,
        }
    }
    with open(os.path.join(args.output_dir, 'regression_cost.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Saved regression_cost.json', summary)

    # 追加写入可读文本公式，便于存档引用
    txt_path = os.path.join(args.output_dir, 'regression_cost.txt')
    with open(txt_path, 'a') as ftxt:
        ftxt.write('==============================\n')
        ftxt.write(f'Total samples: {len(T)}, train: {len(train_idx)}, test: {len(test_idx)}\n')
        ftxt.write(f'T_step(ms) = {coef_T[0]:.6f}*F(G) + {coef_T[1]:.6f}*P(M) + {coef_T[2]:.6f}*(1/B) + {coef_T[3]:.6f}\n')
        ftxt.write(f'  R2_train={r2_Ttr:.4f}  R2_test={r2_Tte:.4f}  MSE_train={mse_Ttr:.4f}  MSE_test={mse_Tte:.4f}\n')
        ftxt.write(f'M_max(MB) = {coef_M[0]:.6f}*F(G) + {coef_M[1]:.6f}*P(M) + {coef_M[2]:.6f}*B + {coef_M[3]:.6f}\n')
        ftxt.write(f'  R2_train={r2_Mtr:.4f}  R2_test={r2_Mte:.4f}  MSE_train={mse_Mtr:.4f}  MSE_test={mse_Mte:.4f}\n')
    print('Appended formulas to', txt_path)

    # 可视化
    # F vs T/M with P as color
    F = X_T[:,0]; Pval = X_T[:,1]; B = X_M[:,2]
    plot_scatter_color(F, T, Pval, 'F (GFLOPs)', 'T_step (ms)', 'P (M)',
                       os.path.join(args.output_dir, 'F_T_colorP.png'), clean=False)
    plot_scatter_color(F, M, Pval, 'F (GFLOPs)', 'M_max (MB)', 'P (M)',
                       os.path.join(args.output_dir, 'F_M_colorP.png'), clean=False)
    plot_scatter_color(F, T, Pval, '','', '', os.path.join(args.output_dir, 'F_T_colorP_clean.png'), clean=True)
    plot_scatter_color(F, M, Pval, '','', '', os.path.join(args.output_dir, 'F_M_colorP_clean.png'), clean=True)

    # 3D 三元散点：F, P, 1/B -> T；F, P, B -> M
    invB = 1.0 / np.array([float(r['batch_size']) for r in rows])
    Bval = np.array([float(r['batch_size']) for r in rows])
    plot_3d_scatter(F, Pval, invB, T, xlabel='F (G)', ylabel='P (M)', zlabel='1/B',
                    clabel='T_step (ms)', save_path=os.path.join(args.output_dir, 'F_P_invB_T_3d.png'),
                    clean=False, cmap='plasma')
    plot_3d_scatter(F, Pval, invB, T, xlabel='', ylabel='', zlabel='',
                    clabel='', save_path=os.path.join(args.output_dir, 'F_P_invB_T_3d_clean.png'),
                    clean=True, cmap='plasma')
    plot_3d_scatter(F, Pval, Bval, M, xlabel='F (G)', ylabel='P (M)', zlabel='B',
                    clabel='M_max (MB)', save_path=os.path.join(args.output_dir, 'F_P_B_M_3d.png'),
                    clean=False, cmap='magma')
    plot_3d_scatter(F, Pval, Bval, M, xlabel='', ylabel='', zlabel='',
                    clabel='', save_path=os.path.join(args.output_dir, 'F_P_B_M_3d_clean.png'),
                    clean=True, cmap='magma')

    # pred vs true & hist (test set)
    plot_pred_true(Tte, pred_Tte, os.path.join(args.output_dir, 'pred_true_T.png'), 'T_step pred vs true', clean=False)
    plot_pred_true(Tte, pred_Tte, os.path.join(args.output_dir, 'pred_true_T_clean.png'), '', clean=True)
    plot_pred_true(Mte, pred_Mte, os.path.join(args.output_dir, 'pred_true_M.png'), 'M_max pred vs true', clean=False)
    plot_pred_true(Mte, pred_Mte, os.path.join(args.output_dir, 'pred_true_M_clean.png'), '', clean=True)

    rel_err_T = (pred_Tte - Tte) / Tte
    rel_err_M = (pred_Mte - Mte) / Mte
    plot_hist(rel_err_T, os.path.join(args.output_dir, 'relerr_T.png'), 'RelErr T', clean=False)
    plot_hist(rel_err_T, os.path.join(args.output_dir, 'relerr_T_clean.png'), '', clean=True)
    plot_hist(rel_err_M, os.path.join(args.output_dir, 'relerr_M.png'), 'RelErr M', clean=False)
    plot_hist(rel_err_M, os.path.join(args.output_dir, 'relerr_M_clean.png'), '', clean=True)


if __name__ == '__main__':
    main()
