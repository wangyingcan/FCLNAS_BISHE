#!/usr/bin/env python3
"""
Task-wise accuracy visualization for 7 baselines + 1 proposed method.

Usage:
    python3 plot_task_accuracy_non_iid.py
    python3 plot_task_accuracy_non_iid.py --data-json /path/to/data.json

Data format (for both inline data and JSON file):
{
  "FedAvg":   {"mean": [10 values]},
  "TARGET":   {"mean": [10 values]},
  "ReFed":    {"mean": [10 values]},
  "AF-FCL":   {"mean": [10 values]},
  "FedTA":    {"mean": [10 values]},
  "Ditto":    {"mean": [10 values], "best": [10 values], "worst": [10 values]},
  "FedWeIT":  {"mean": [10 values], "best": [10 values], "worst": [10 values]},
  "FedCLRas": {"mean": [10 values], "best": [10 values], "worst": [10 values]}
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

NUM_TASKS = 10
METHOD_ORDER = [
    "FedAvg",
    "TARGET",
    "ReFed",
    "AF-FCL",
    "FedTA",
    "Ditto",
    "FedWeIT",
    "FedCLRas",
]
PERSONALIZED_METHODS = {"Ditto", "FedWeIT", "FedCLRas"}

COLOR_MAP = {
    "FedAvg": "#4E79A7",
    "TARGET": "#F28E2B",
    "ReFed": "#59A14F",
    "AF-FCL": "#E15759",
    "FedTA": "#B07AA1",
    "Ditto": "#EDC948",
    "FedWeIT": "#76B7B2",
    "FedCLRas": "#2F2F2F",
}

# =========================
# Manual data entry section
# =========================
# Replace the demo values below with your real 10-task results.
INLINE_DATA: Dict[str, Dict[str, List[float]]] = {
    "FedAvg": {"mean": [63.5, 61.8, 60.9, 59.7, 58.6, 57.8, 56.9, 56.1, 55.4, 54.8]},
    "TARGET": {"mean": [64.2, 62.4, 61.5, 60.1, 58.9, 58.1, 57.2, 56.3, 55.8, 55.2]},
    "ReFed": {"mean": [65.0, 63.5, 62.6, 61.2, 60.3, 59.4, 58.6, 57.8, 57.1, 56.4]},
    "AF-FCL": {"mean": [66.1, 64.8, 63.9, 62.8, 61.9, 61.1, 60.4, 59.7, 59.0, 58.4]},
    "FedTA": {"mean": [64.8, 63.2, 62.3, 61.1, 60.2, 59.3, 58.5, 57.9, 57.2, 56.8]},
    "Ditto": {
        "mean": [61.9, 60.1, 58.8, 57.3, 56.2, 55.1, 54.0, 53.0, 52.0, 51.2],
        "best": [64.1, 62.3, 61.0, 59.4, 58.2, 57.0, 56.1, 55.0, 53.8, 53.0],
        "worst": [59.6, 57.8, 56.3, 55.0, 53.9, 52.7, 51.5, 50.6, 49.6, 48.9],
    },
    "FedWeIT": {
        "mean": [66.0, 64.6, 63.8, 62.7, 61.8, 60.9, 60.1, 59.3, 58.6, 58.0],
        "best": [68.0, 66.7, 65.8, 64.8, 63.9, 63.0, 62.2, 61.3, 60.7, 60.0],
        "worst": [63.8, 62.3, 61.4, 60.2, 59.4, 58.5, 57.7, 56.9, 56.2, 55.5],
    },
    "FedCLRas": {
        "mean": [67.8, 66.5, 65.6, 64.9, 64.3, 63.8, 63.2, 62.6, 62.1, 61.7],
        "best": [70.2, 69.0, 68.1, 67.2, 66.8, 66.2, 65.8, 65.1, 64.7, 64.1],
        "worst": [65.3, 64.1, 63.2, 62.5, 61.9, 61.3, 60.7, 60.1, 59.6, 59.2],
    },
}


def _load_data(data_json: str | None) -> Dict[str, Dict[str, List[float]]]:
    if not data_json:
        return INLINE_DATA
    with open(data_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_vec(name: str, vec: List[float], key: str) -> np.ndarray:
    if len(vec) != NUM_TASKS:
        raise ValueError(f"{name}.{key} length must be {NUM_TASKS}, got {len(vec)}")
    arr = np.array(vec, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name}.{key} contains NaN")
    return arr


def _validate(data: Dict[str, Dict[str, List[float]]]) -> None:
    missing = [m for m in METHOD_ORDER if m not in data]
    if missing:
        raise ValueError(f"Missing methods in data: {missing}")
    for method in METHOD_ORDER:
        item = data[method]
        if "mean" not in item:
            raise ValueError(f"{method} missing required key: mean")
        _check_vec(method, item["mean"], "mean")
        if method in PERSONALIZED_METHODS:
            if "best" not in item or "worst" not in item:
                raise ValueError(
                    f"{method} must contain best/worst for personalized visualization"
                )
            best = _check_vec(method, item["best"], "best")
            worst = _check_vec(method, item["worst"], "worst")
            if np.any(best < worst):
                raise ValueError(f"{method}: best should be >= worst at each task")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "STSong"],
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def plot_task_accuracy(
    data: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
    file_stem: str,
    title: str,
) -> None:
    _set_style()
    _validate(data)

    x = np.arange(1, NUM_TASKS + 1)
    fig, ax = plt.subplots(figsize=(12.6, 7.2))

    line_handles = []
    all_vals = []

    for method in METHOD_ORDER:
        color = COLOR_MAP[method]
        mean = np.array(data[method]["mean"], dtype=float)
        all_vals.extend(mean.tolist())
        (line,) = ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=4.6,
            label=method,
            alpha=0.98,
        )
        line_handles.append(line)

        if method in PERSONALIZED_METHODS:
            best = np.array(data[method]["best"], dtype=float)
            worst = np.array(data[method]["worst"], dtype=float)
            all_vals.extend(best.tolist())
            all_vals.extend(worst.tolist())
            ax.scatter(
                x,
                best,
                marker="^",
                s=54,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.95,
                zorder=4,
            )
            ax.scatter(
                x,
                worst,
                marker="v",
                s=54,
                color=color,
                edgecolors="white",
                linewidths=0.7,
                alpha=0.95,
                zorder=4,
            )

    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad = max(2.0, (y_max - y_min) * 0.12)
    ax.set_ylim(np.floor((y_min - pad) / 2) * 2, np.ceil((y_max + pad) / 2) * 2)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in x])
    ax.set_title(title, pad=12)

    ax.grid(axis="y", linestyle="--", linewidth=0.85, alpha=0.35)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_methods = ax.legend(
        handles=line_handles,
        title="Method (Task Mean Accuracy)",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    ax.add_artist(legend_methods)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="None",
            markersize=7,
            label="Best client (personalized methods only)",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="gray",
            linestyle="None",
            markersize=7,
            label="Worst client (personalized methods only)",
        ),
    ]
    ax.legend(
        handles=marker_handles,
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(1.0, 1.02),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{file_stem}.png"
    pdf_path = out_dir / f"{file_stem}.pdf"
    fig.savefig(png_path, dpi=420, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Figure saved: {png_path}")
    print(f"[OK] Figure saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot task-wise accuracy comparison figure.")
    parser.add_argument(
        "--data-json",
        type=str,
        default=None,
        help="Optional JSON file path. If not provided, INLINE_DATA in this script is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="figures",
        help="Output directory for figure files.",
    )
    parser.add_argument(
        "--file-stem",
        type=str,
        default="task_accuracy_non_iid_alpha03",
        help="Output filename stem (without extension).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="CIFAR-100 non-IID (alpha=0.3): Task-wise Accuracy Over 10 Tasks",
        help="Figure title.",
    )
    args = parser.parse_args()

    data = _load_data(args.data_json)
    plot_task_accuracy(
        data=data,
        out_dir=Path(args.out_dir),
        file_stem=args.file_stem,
        title=args.title,
    )


if __name__ == "__main__":
    main()
