import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["feat"], data["label"], data["task_id"], data["backbone"]


def prepare_tsne(feat, do_pca=True, pca_dim=50, perplexity=30, random_state=0):
    feat = StandardScaler().fit_transform(feat)
    if do_pca and feat.shape[1] > pca_dim:
        feat = PCA(n_components=pca_dim, random_state=random_state).fit_transform(feat)
    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    emb = tsne.fit_transform(feat)
    return emb


def plot_backbones(embeds, colors, titles, out_file, cmap="tab10", s=5, alpha=0.7):
    n = len(embeds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        vals = np.unique(colors[i])
        # 将离散任务/类别值映射到 0..K-1，保证颜色与刻度一一对应
        idx_map = {v: k for k, v in enumerate(sorted(vals))}
        color_idx = np.vectorize(idx_map.get)(colors[i])
        cmap_sub = plt.get_cmap(cmap, len(vals))
        norm = mcolors.Normalize(vmin=0, vmax=len(vals) - 1)
        sc = ax.scatter(embeds[i][:, 0], embeds[i][:, 1], c=color_idx, cmap=cmap_sub, norm=norm, s=s, alpha=alpha)
        # 去掉顶部 title，保留坐标刻度
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(labelsize=8)
        if i == n - 1:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            tick_pos = np.arange(len(vals))
            cbar.set_ticks(tick_pos)
            cbar.set_ticklabels([str(i + 1) for i in range(len(vals))])
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[t-SNE] saved {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_files", nargs="+", required=True, help="三个 npz 路径，按 backbone 顺序给出")
    ap.add_argument("--by", type=str, default="task", choices=["task", "class"], help="上色方式：task 或 class")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--no_pca", action="store_true", help="关闭 PCA 预降维")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--out_dir", type=str, default="./cil_tsne_outputs", help="图片输出目录")
    ap.add_argument("--out_prefix", type=str, default="tsne_backbone_compare", help="输出文件名前缀")
    ap.add_argument("--marker_size", type=float, default=7.0, help="散点大小")
    ap.add_argument("--alpha", type=float, default=0.7, help="散点透明度")
    ap.add_argument("--max_points", type=int, default=None, help="最多使用多少样本做 t-SNE（多余随机下采样）")
    ap.add_argument("--max_classes", type=int, default=None, help="by class 时随机保留最多多少个类别（如 CIFAR100 抽 20 类）")
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats_all, colors_all, titles = [], [], []
    for f in args.feature_files:
        feat, label, task, backbone = load_npz(f)
        # by class 可选随机抽取若干类
        if args.by == "class" and args.max_classes is not None:
            rng = random.Random(args.random_state)
            uniq_cls = list(np.unique(label))
            rng.shuffle(uniq_cls)
            keep_cls = set(uniq_cls[: args.max_classes])
            keep_idx = [i for i, c in enumerate(label) if c in keep_cls]
            feat = feat[keep_idx]
            label = label[keep_idx]
            task = task[keep_idx]
        if args.max_points is not None and feat.shape[0] > args.max_points:
            idx = list(range(feat.shape[0]))
            random.Random(args.random_state).shuffle(idx)
            idx = idx[: args.max_points]
            feat = feat[idx]
            label = label[idx]
            task = task[idx]
        emb = prepare_tsne(
            feat,
            do_pca=not args.no_pca,
            pca_dim=args.pca_dim,
            perplexity=args.perplexity,
            random_state=args.random_state,
        )
        if args.by == "task":
            color = task
        else:
            color = label
        feats_all.append(emb)
        colors_all.append(color)
        titles.append(str(backbone[0]))

    suffix = "task" if args.by == "task" else "class"
    out_file = os.path.join(args.out_dir, f"{args.out_prefix}_by_{suffix}.png")
    cmap = "tab10" if args.by == "task" else "tab20"
    plot_backbones(feats_all, colors_all, titles, out_file, cmap=cmap, s=args.marker_size, alpha=args.alpha)


if __name__ == "__main__":
    main()
