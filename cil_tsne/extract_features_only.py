"""
单独的特征导出脚本：不用重新训练，直接从已有 checkpoint 抽取倒数第二层特征，支持放大 samples_per_class。
示例：
python3 cil_tsne/extract_features_only.py \
  --backbone resnet18 \
  --checkpoint cil_tsne_outputs/checkpoints/cil_resnet18_final.pth \
  --dataset CIFAR10 \
  --dataset_location /home/wyc/data \
  --num_tasks 2 --classes_per_task 5 \
  --samples_per_class 200 \
  --out features_resnet18_more.npz
"""

import argparse
import os
import sys

# 保证能 import 现有工具
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cil_tsne.run_cil_feature_tsne import (  # noqa: E402
    build_task_indices,
    extract_features_for_backbone,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, required=True, help="resnet18 / small_model / proxyless_subnet ...")
    ap.add_argument("--checkpoint", type=str, required=True, help="已训练好的 state_dict ckpt 路径")
    ap.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    ap.add_argument("--dataset_location", type=str, required=True)
    ap.add_argument("--num_tasks", type=int, required=True)
    ap.add_argument("--classes_per_task", type=int, required=True)
    ap.add_argument("--samples_per_class", type=int, default=200, help="每类抽取多少张用于特征")
    ap.add_argument("--train_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--proxyless_config", type=str, default=None, help="仅 proxyless_subnet 需要")
    ap.add_argument("--out", type=str, required=True, help="输出 npz 路径")
    ap.add_argument("--save_taskwise_features", action="store_true", help="额外按 task 输出 features_*_taskN.npz")
    return ap.parse_args()


def main():
    args = parse_args()
    num_classes = 10 if args.dataset.lower() == "cifar10" else 100
    if args.num_tasks * args.classes_per_task != num_classes:
        raise ValueError(f"{args.dataset}: num_tasks * classes_per_task 必须等于 {num_classes}")
    tasks = build_task_indices(num_classes=num_classes, num_tasks=args.num_tasks, classes_per_task=args.classes_per_task, seed=0)
    extract_features_for_backbone(
        backbone_name=args.backbone,
        checkpoint_path=args.checkpoint,
        feature_save_path=args.out,
        dataset_root=args.dataset_location,
        tasks=tasks,
        samples_per_class=args.samples_per_class,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        proxyless_config=args.proxyless_config,
        dataset=args.dataset,
        save_taskwise=args.save_taskwise_features,
    )


if __name__ == "__main__":
    main()
