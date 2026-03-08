#!/usr/bin/env python3
from __future__ import annotations
"""
Plot per-client label histograms for task-1 under:
1) IID split
2) Non-IID Dirichlet split (alpha=0.3 by default)

This script reuses the same split logic as training by calling FCLDataManager.
"""

import argparse
import json
import os
from typing import Dict, List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data_providers.cifar100_fcl_dirichlet_split import FCLDataManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot task-1 client label histograms for IID and Non-IID splits."
    )
    parser.add_argument("--dataset_location", type=str, required=True, help="Path to CIFAR-100 data root.")
    parser.add_argument("--num_users", type=int, default=10, help="Number of clients.")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks.")
    parser.add_argument("--task_id", type=int, default=1, help="Task id to visualize (default: 1).")
    parser.add_argument("--seed", type=int, default=0, help="Split random seed.")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha for non-IID case.")
    parser.add_argument(
        "--include_val",
        action="store_true",
        help="If set, use train+val samples for each client histogram (default: train only).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./analysis/label_hist_task1",
        help="Output directory for figures/json files.",
    )
    return parser.parse_args()


def load_targets(dataset_location: str) -> Tuple[np.ndarray, np.ndarray]:
    import torchvision.datasets as datasets

    train_set = datasets.CIFAR100(root=dataset_location, train=True, download=False)
    test_set = datasets.CIFAR100(root=dataset_location, train=False, download=False)
    return np.asarray(train_set.targets), np.asarray(test_set.targets)


def build_manager(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    num_users: int,
    num_tasks: int,
    alpha: float,
    val_ratio: float,
    seed: int,
    iid: bool,
) -> FCLDataManager:
    from data_providers.cifar100_fcl_dirichlet_split import FCLDataManager

    return FCLDataManager(
        train_targets=train_targets,
        test_targets=test_targets,
        num_clients=num_users,
        num_tasks=num_tasks,
        classes_per_task=None,
        alpha=alpha,
        val_ratio=val_ratio,
        seed=seed,
        precompute=False,
        iid=iid,
    )


def collect_counts(
    manager: FCLDataManager,
    train_targets: np.ndarray,
    task_id: int,
    include_val: bool,
) -> Tuple[List[int], Dict[int, np.ndarray]]:
    task_classes = list(manager._task_classes[task_id])
    client_counts: Dict[int, np.ndarray] = {}
    for cid in range(manager.K):
        trn_idx, val_idx, _ = manager.get(client_id=cid, task_id=task_id)
        if include_val:
            used_idx = np.asarray(trn_idx + val_idx, dtype=int)
        else:
            used_idx = np.asarray(trn_idx, dtype=int)
        if used_idx.size == 0:
            counts = np.zeros(100, dtype=int)
        else:
            labels = train_targets[used_idx]
            counts = np.bincount(labels, minlength=100)
        client_counts[cid] = counts
    return task_classes, client_counts


def save_json(
    out_file: str,
    mode_name: str,
    task_id: int,
    task_classes: List[int],
    client_counts: Dict[int, np.ndarray],
):
    payload = {
        "mode": mode_name,
        "task_id": int(task_id),
        "task_classes": [int(c) for c in task_classes],
        "clients": {},
    }
    for cid, counts in client_counts.items():
        payload["clients"][str(cid)] = {
            "num_samples": int(np.sum(counts)),
            "counts_all_100_classes": [int(v) for v in counts.tolist()],
            "counts_task_classes": {str(c): int(counts[c]) for c in task_classes},
        }
    with open(out_file, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def save_figure(
    out_file: str,
    mode_title: str,
    task_id: int,
    task_classes: List[int],
    client_counts: Dict[int, np.ndarray],
    include_val: bool,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            f"matplotlib import failed: {exc}. Please install matplotlib in current env."
        )

    n_clients = len(client_counts)
    nrows, ncols = 2, 5
    if n_clients != 10:
        ncols = min(5, max(1, n_clients))
        nrows = int(np.ceil(n_clients / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(task_classes))
    xticklabels = [str(c) for c in task_classes]

    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n_clients:
            ax.axis("off")
            continue
        counts = client_counts[i]
        y = np.asarray([counts[c] for c in task_classes], dtype=int)
        ax.bar(x, y)
        ax.set_title(f"Client {i} (n={int(np.sum(y))})")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=8)
        ax.set_xlabel("Class ID")
        ax.set_ylabel("Count")

    data_scope = "train+val" if include_val else "train"
    fig.suptitle(f"{mode_title} | Task {task_id} | {data_scope} label histogram", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_file, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_targets, test_targets = load_targets(args.dataset_location)

    run_settings = [
        ("iid", True, 1.0),
        (f"noniid_alpha_{args.alpha}", False, args.alpha),
    ]

    for mode_name, iid_flag, alpha_value in run_settings:
        manager = build_manager(
            train_targets=train_targets,
            test_targets=test_targets,
            num_users=args.num_users,
            num_tasks=args.num_tasks,
            alpha=alpha_value,
            val_ratio=args.val_ratio,
            seed=args.seed,
            iid=iid_flag,
        )
        task_classes, client_counts = collect_counts(
            manager=manager,
            train_targets=train_targets,
            task_id=args.task_id,
            include_val=args.include_val,
        )

        fig_path = os.path.join(args.output_dir, f"{mode_name}_task{args.task_id}_client_label_hist.png")
        json_path = os.path.join(args.output_dir, f"{mode_name}_task{args.task_id}_client_label_hist.json")
        save_figure(
            out_file=fig_path,
            mode_title=mode_name,
            task_id=args.task_id,
            task_classes=task_classes,
            client_counts=client_counts,
            include_val=args.include_val,
        )
        save_json(
            out_file=json_path,
            mode_name=mode_name,
            task_id=args.task_id,
            task_classes=task_classes,
            client_counts=client_counts,
        )
        print(f"[OK] saved figure: {fig_path}")
        print(f"[OK] saved counts: {json_path}")


if __name__ == "__main__":
    main()
