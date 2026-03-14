#!/usr/bin/env python3
"""
Task-wise model complexity trend (FLOPs) for the proposed method:
- core line 1: IID mean FLOPs over tasks
- core line 2: non-IID mean FLOPs over tasks
- points for per-task max/min FLOPs in each setting

Usage:
    python3 plot_task_flops_trend.py
    python3 plot_task_flops_trend.py --data-json /path/to/flops_data.json
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

# ===========================================
# Manual data entry section (edit this block)
# ===========================================
# Unit: MFLOPs
INLINE_DATA: Dict[str, Dict[str, List[float]]] = {
    "iid": {
        "mean": [288, 292, 295, 300, 302, 305, 309, 312, 316, 320],
        "max": [304, 307, 311, 316, 319, 322, 326, 329, 334, 339],
        "min": [271, 276, 279, 283, 286, 288, 293, 296, 298, 304],
    },
    "non_iid": {
        "mean": [303, 307, 311, 315, 318, 322, 327, 331, 336, 341],
        "max": [322, 326, 331, 336, 339, 344, 349, 354, 359, 366],
        "min": [286, 291, 294, 299, 302, 305, 311, 315, 319, 324],
    },
}


def _load_data(data_json: str | None) -> Dict[str, Dict[str, List[float]]]:
    if not data_json:
        return INLINE_DATA
    with open(data_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_arr(name: str, key: str, values: List[float]) -> np.ndarray:
    if len(values) != NUM_TASKS:
        raise ValueError(f"{name}.{key} length must be {NUM_TASKS}, got {len(values)}")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name}.{key} contains NaN")
    return arr


def _validate_data(data: Dict[str, Dict[str, List[float]]]) -> None:
    for setting in ("iid", "non_iid"):
        if setting not in data:
            raise ValueError(f"Missing required key: {setting}")
        section = data[setting]
        for key in ("mean", "max", "min"):
            if key not in section:
                raise ValueError(f"Missing required key: {setting}.{key}")
        mean = _check_arr(setting, "mean", section["mean"])
        max_v = _check_arr(setting, "max", section["max"])
        min_v = _check_arr(setting, "min", section["min"])
        if np.any(max_v < mean):
            raise ValueError(f"{setting}: max must be >= mean at each task")
        if np.any(mean < min_v):
            raise ValueError(f"{setting}: mean must be >= min at each task")


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


def plot_flops_trend(
    data: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
    file_stem: str,
    title: str,
) -> None:
    _set_style()
    _validate_data(data)

    x = np.arange(1, NUM_TASKS + 1)
    fig, ax = plt.subplots(figsize=(12.2, 6.8))

    style_map = {
        "iid": {"color": "#2C7FB8", "label": "Proposed Method (IID)"},
        "non_iid": {"color": "#D95F0E", "label": "Proposed Method (non-IID, alpha=0.3)"},
    }

    all_vals = []
    line_handles = []

    for setting in ("iid", "non_iid"):
        color = style_map[setting]["color"]
        label = style_map[setting]["label"]
        mean = np.asarray(data[setting]["mean"], dtype=float)
        max_v = np.asarray(data[setting]["max"], dtype=float)
        min_v = np.asarray(data[setting]["min"], dtype=float)

        all_vals.extend(mean.tolist())
        all_vals.extend(max_v.tolist())
        all_vals.extend(min_v.tolist())

        # Range stem per task to show complexity span (min->max)
        ax.vlines(
            x,
            min_v,
            max_v,
            colors=color,
            linestyles="-",
            linewidth=1.1,
            alpha=0.22,
            zorder=1,
        )

        (line,) = ax.plot(
            x,
            mean,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=5.0,
            label=label,
            zorder=3,
        )
        line_handles.append(line)

        ax.scatter(
            x,
            max_v,
            marker="^",
            s=56,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            zorder=4,
        )
        ax.scatter(
            x,
            min_v,
            marker="v",
            s=56,
            color=color,
            edgecolors="white",
            linewidths=0.7,
            alpha=0.95,
            zorder=4,
        )

    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    pad = max(8.0, (y_max - y_min) * 0.12)
    ax.set_ylim(np.floor((y_min - pad) / 5) * 5, np.ceil((y_max + pad) / 5) * 5)

    ax.set_xlabel("Task ID")
    ax.set_ylabel("Model Complexity (MFLOPs)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in x])
    ax.set_title(title, pad=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.85, alpha=0.35)
    ax.grid(axis="x", linestyle="-", linewidth=0.4, alpha=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_main = ax.legend(
        handles=line_handles,
        loc="upper left",
        frameon=False,
        title="Task Mean FLOPs",
    )
    ax.add_artist(legend_main)

    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="gray",
            linestyle="None",
            markersize=7,
            label="Task Max FLOPs",
        ),
        Line2D(
            [0],
            [0],
            marker="v",
            color="gray",
            linestyle="None",
            markersize=7,
            label="Task Min FLOPs",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1.3,
            alpha=0.35,
            label="Task FLOPs range",
        ),
    ]
    ax.legend(
        handles=marker_handles,
        loc="upper right",
        frameon=False,
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
    parser = argparse.ArgumentParser(
        description="Plot task-wise FLOPs trend (IID vs non-IID) for the proposed method."
    )
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
        help="Output directory for PNG/PDF.",
    )
    parser.add_argument(
        "--file-stem",
        type=str,
        default="task_flops_trend_iid_vs_non_iid",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Proposed Method: Task-wise FLOPs Trend in IID vs non-IID",
        help="Figure title.",
    )
    args = parser.parse_args()

    data = _load_data(args.data_json)
    plot_flops_trend(
        data=data,
        out_dir=Path(args.out_dir),
        file_stem=args.file_stem,
        title=args.title,
    )


if __name__ == "__main__":
    main()
