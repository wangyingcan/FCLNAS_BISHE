import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["feat"], data["label"], data["task_id"], data["backbone"]


def prepare_tsne(feat, do_pca=True, pca_dim=50, random_state=0):
    feat = StandardScaler().fit_transform(feat)
    if do_pca and feat.shape[1] > pca_dim:
        feat = PCA(n_components=pca_dim, random_state=random_state).fit_transform(feat)
    tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
    emb = tsne.fit_transform(feat)
    return emb


def plot_backbones(embeds, colors, titles, out_file, cmap="tab10"):
    n = len(embeds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        sc = ax.scatter(embeds[i][:, 0], embeds[i][:, 1], c=colors[i], cmap=cmap, s=5, alpha=0.7)
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])
        if i == n - 1:
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"[t-SNE] saved {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_files", nargs="+", required=True, help="三个 npz 路径，按 backbone 顺序给出")
    ap.add_argument("--by", type=str, default="task", choices=["task", "class"], help="上色方式：task 或 class")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="./cil_tsne_outputs", help="图片输出目录")
    ap.add_argument("--out_prefix", type=str, default="tsne_backbone_compare", help="输出文件名前缀")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats_all, colors_all, titles = [], [], []
    for f in args.feature_files:
        feat, label, task, backbone = load_npz(f)
        emb = prepare_tsne(feat, do_pca=True, pca_dim=args.pca_dim, random_state=0)
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
    plot_backbones(feats_all, colors_all, titles, out_file, cmap=cmap)


if __name__ == "__main__":
    main()
