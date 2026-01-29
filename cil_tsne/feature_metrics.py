import argparse
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["feat"], data["label"], data["task_id"], data["backbone"]


def prep_feat(feat, do_pca=True, pca_dim=50, random_state=0):
    feat = StandardScaler().fit_transform(feat)
    if do_pca and feat.shape[1] > pca_dim:
        feat = PCA(n_components=pca_dim, random_state=random_state).fit_transform(feat)
    return feat


def class_distances(feat, label):
    """返回类内平均距离、类间中心距离平均值"""
    label = np.asarray(label)
    uniq = np.unique(label)
    # 类中心
    centers = {}
    intra = []
    for c in uniq:
        idx = label == c
        f = feat[idx]
        centers[c] = f.mean(axis=0)
        if f.shape[0] > 1:
            dmat = pairwise_distances(f)
            intra.append(dmat[np.triu_indices_from(dmat, k=1)].mean())
    intra_mean = float(np.mean(intra)) if intra else np.nan
    inter = []
    for i, ci in enumerate(uniq):
        for cj in uniq[i + 1 :]:
            inter.append(np.linalg.norm(centers[ci] - centers[cj]))
    inter_mean = float(np.mean(inter)) if inter else np.nan
    return intra_mean, inter_mean


def knn_probe(feat, label, k=5, random_state=0):
    """简单的 kNN 线性探针：半数训练，半数测试（分层采样）"""
    rng = random.Random(random_state)
    feat = np.asarray(feat)
    label = np.asarray(label)
    train_idx, test_idx = [], []
    for c in np.unique(label):
        idx = np.where(label == c)[0].tolist()
        rng.shuffle(idx)
        mid = len(idx) // 2
        train_idx.extend(idx[:mid])
        test_idx.extend(idx[mid:])
    X_train, y_train = feat[train_idx], label[train_idx]
    X_test, y_test = feat[test_idx], label[test_idx]
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_files", nargs="+", required=True, help="npz 特征文件列表")
    ap.add_argument("--no_pca", action="store_true")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--knn_k", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=0)
    args = ap.parse_args()

    for f in args.feature_files:
        feat, label, task, backbone = load_npz(f)
        name = str(backbone[0])
        feat_p = prep_feat(
            feat, do_pca=not args.no_pca, pca_dim=args.pca_dim, random_state=args.random_state
        )
        # 全量类别 silhouette
        sil = silhouette_score(feat_p, label) if len(np.unique(label)) > 1 else np.nan
        intra, inter = class_distances(feat_p, label)
        sep = inter / intra if (intra and intra > 0 and inter) else np.nan
        knn_acc = knn_probe(feat_p, label, k=args.knn_k, random_state=args.random_state)
        print(
            f"[Metric] {name} | silhouette={sil:.4f} | intra={intra:.4f} | inter={inter:.4f} | inter/intra={sep:.4f} | kNN@{args.knn_k}={knn_acc:.4f}"
        )


if __name__ == "__main__":
    main()
