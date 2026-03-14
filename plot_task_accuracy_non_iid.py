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
from matplotlib import font_manager
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

LINESTYLE_MAP = {
    "FedAvg": "-",
    "TARGET": "-",
    "ReFed": "-",
    "AF-FCL": "-",
    "FedTA": "-",
    "Ditto": "-",
    "FedWeIT": "-",
    "FedCLRas": "-",
}

MARKER_MAP = {
    "FedAvg": "o",
    "TARGET": "s",
    "ReFed": "D",
    "AF-FCL": "P",
    "FedTA": "X",
    "Ditto": "h",
    "FedWeIT": "^",
    "FedCLRas": "*",
}

# =========================
# Manual data entry section
# =========================
# Replace the demo values below with your real 10-task results.
INLINE_DATA: Dict[str, Dict[str, List[float]]] = {
    "FedAvg": {"mean": [12.5,13.9,26.5,21,36,32.9,34.9,49.9,35,39.7]},
    "TARGET": {"mean": [16.5,9.6,27.1,27.4,42.5,36.1,39.6,53.3,38.8,45.9]},
    "ReFed": {"mean": [15,15.2,24.1,22.7,37.7,35,36.1,51.6,38.8,44.1]},
    "AF-FCL": {"mean": [14.2,15.1,26.9,24.3,40.8,38.2,40.7,54.8,41.2,52.6]},
    "FedTA": {"mean": [13.2,15.3,25.7,22.6,38.4,34.9,36.5,51.3,37.6,40.7]},
    "Ditto": {
        "mean": [13.13,14.36,15.86,15.76,20.08,20.25,22.68,29.85,23.6,29.23],
        "best": [40.7,39.2,44.5,51.6,43.3,38.7,35.9,49.9,31,42.5],
        "worst": [10.0,10.2,11.2,10.8,15.7,15.2,17.8,19.7,12.7,16.8],
    },
    "FedWeIT": {
        "mean": [13.1,17.8,23.3,20.6,36.8,35.6,38.2,50.3,40.5,41.4],
        # "best": [14.04,19.44,24.79,22.09,38.18,36.5,38.47,50.35,40.77,41.58],
        # "worst": [12.04,16.14,22.09,19.29,35.88,34.8,38.07,50.15,40.17,41.08],
    },
    
    "FedCLRas": {
        "mean": [33.1,37.05,39.89,30.2,43.68,40.84,33.76,39.36,29.15,42.31],
        "best": [39.7,39.8,41.1,32.7,46.8,43.7,35.6,41.8,30.6,45.2],
        "worst": [27.6,32.3,38.7,27.1,40.8,38.9,32.5,37.4,27.8,40.2],
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
            has_best = "best" in item
            has_worst = "worst" in item
            if has_best != has_worst:
                raise ValueError(f"{method} must provide both best and worst, or neither")
            if has_best and has_worst:
                _check_vec(method, item["best"], "best")
                _check_vec(method, item["worst"], "worst")


def _set_style() -> None:
    cjk_candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    picked = next((name for name in cjk_candidates if name in available), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [picked, "Times New Roman", "DejaVu Sans"],
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
    fig, ax = plt.subplots(figsize=(14.2, 7.2))
    fig.subplots_adjust(right=0.78, top=0.88)

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
            linestyle=LINESTYLE_MAP[method],
            marker=MARKER_MAP[method],
            markersize=4.6,
            label=method,
            alpha=0.98,
        )
        line_handles.append(line)

        if method in PERSONALIZED_METHODS and "best" in data[method] and "worst" in data[method]:
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

    ax.set_xlabel("")
    ax.set_ylabel("精度 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"任务{i}" for i in x])
    # 用户要求：不显示图标题

    ax.grid(axis="y", linestyle="--", linewidth=0.85, alpha=0.35)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_methods = ax.legend(
        handles=line_handles,
        title="方法（任务平均精度）",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.6,
        borderaxespad=0.0,
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
            label="最优客户端（仅个性化方法）",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="gray",
            linestyle="None",
            markersize=7,
            label="最差客户端（仅个性化方法）",
        ),
    ]
    ax.legend(
        handles=marker_handles,
        loc="upper left",
        frameon=False,
        bbox_to_anchor=(1.01, 0.45),
        borderaxespad=0.0,
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
        default="CIFAR-100 非IID（α=0.3）10任务精度曲线",
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
