#!/usr/bin/env python3
"""
Collect per-task per-client FLOPs from net_info.txt files.

Expected directory shape (example):
  <exp_root>/
    xxx-task1/
      learned_net/clients/client_0/net_info.txt
      learned_net/clients/client_1/net_info.txt
      ...
    xxx-task2/
      learned_net/clients/client_0/net_info.txt
      ...

The script will also try:
  - learned_net/clients/client_x/logs/net_info.txt
  - clients/client_x/net_info.txt
  - clients/client_x/logs/net_info.txt

Outputs:
  1) task_client_flops.csv  (per task per client detail)
  2) task_flops_summary.csv (per task mean/max/min + GLOBAL row)
  3) task_flops_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Dict, List, Optional, Tuple


TASK_DIR_RE = re.compile(r".*-task(?P<task>\d+)$")
CLIENT_DIR_RE = re.compile(r"client_(?P<cid>\d+)$")
FLOPS_LINE_RE = re.compile(r'"flops"\s*:\s*"(?P<val>[^"]+)"', re.IGNORECASE)
NUM_UNIT_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*([kKmMbB]?)\s*$")


@dataclass
class ClientFlops:
    task_id: int
    task_dir: Path
    client_id: int
    client_dir: Path
    net_info_path: Path
    flops_m: float


def parse_metric_to_million(text: str) -> Optional[float]:
    """Convert values like 341.2M / 123K / 1.8B / 341200000 to million units."""
    if text is None:
        return None
    s = str(text).strip()
    m = NUM_UNIT_RE.match(s)
    if m is None:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "B":
        return value * 1000.0
    if unit == "K":
        return value / 1000.0
    if unit == "M":
        return value
    # no unit: assume raw count if very large
    return value / 1_000_000.0 if abs(value) > 100_000 else value


def load_flops_from_net_info(path: Path) -> Optional[float]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not text:
        return None

    # Case 1: strict JSON
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "flops" in payload:
            return parse_metric_to_million(str(payload["flops"]))
    except Exception:
        pass

    # Case 2: regex fallback
    m = FLOPS_LINE_RE.search(text)
    if m is not None:
        return parse_metric_to_million(m.group("val"))

    # Case 3: loose fallback from first numeric token with unit
    candidates = re.findall(r"[+-]?\d+(?:\.\d+)?\s*[kKmMbB]?", text)
    for token in candidates:
        val = parse_metric_to_million(token)
        if val is not None:
            return val
    return None


def discover_task_dirs(exp_root: Path) -> List[Tuple[int, Path]]:
    # Case 0: exp_root itself is a task directory.
    m_root = TASK_DIR_RE.match(exp_root.name)
    if m_root is not None:
        return [(int(m_root.group("task")), exp_root)]

    direct = []
    for p in exp_root.iterdir():
        if not p.is_dir():
            continue
        m = TASK_DIR_RE.match(p.name)
        if m is None:
            continue
        direct.append((int(m.group("task")), p))
    if direct:
        return sorted(direct, key=lambda x: x[0])

    # fallback: recursive search
    recursive = []
    for p in exp_root.rglob("*-task*"):
        if not p.is_dir():
            continue
        m = TASK_DIR_RE.match(p.name)
        if m is None:
            continue
        recursive.append((int(m.group("task")), p))
    # Remove duplicates while keeping smallest path (stable order by task then path)
    unique: Dict[Tuple[int, str], Tuple[int, Path]] = {}
    for tid, p in sorted(recursive, key=lambda x: (x[0], str(x[1]))):
        unique[(tid, str(p.resolve()))] = (tid, p)
    return sorted(unique.values(), key=lambda x: (x[0], str(x[1])))


def discover_client_dirs(task_dir: Path) -> List[Tuple[int, Path]]:
    candidates = [task_dir / "learned_net" / "clients", task_dir / "clients"]
    for root in candidates:
        if not root.is_dir():
            continue
        out = []
        for p in root.iterdir():
            if not p.is_dir():
                continue
            m = CLIENT_DIR_RE.match(p.name)
            if m is None:
                continue
            out.append((int(m.group("cid")), p))
        if out:
            return sorted(out, key=lambda x: x[0])
    return []


def find_net_info(client_dir: Path) -> Optional[Path]:
    for rel in ("net_info.txt", "logs/net_info.txt"):
        p = client_dir / rel
        if p.is_file():
            return p
    return None


def collect_all(exp_root: Path) -> Tuple[List[ClientFlops], List[str]]:
    records: List[ClientFlops] = []
    warnings: List[str] = []

    task_dirs = discover_task_dirs(exp_root)
    if not task_dirs:
        warnings.append(f"[WARN] no task dirs found under: {exp_root}")
        return records, warnings

    for task_id, task_dir in task_dirs:
        clients = discover_client_dirs(task_dir)
        if not clients:
            warnings.append(f"[WARN] task{task_id}: no client dirs in {task_dir}")
            continue
        for cid, cdir in clients:
            info_path = find_net_info(cdir)
            if info_path is None:
                warnings.append(f"[WARN] task{task_id} client_{cid}: net_info.txt not found")
                continue
            flops_m = load_flops_from_net_info(info_path)
            if flops_m is None:
                warnings.append(f"[WARN] task{task_id} client_{cid}: cannot parse flops in {info_path}")
                continue
            records.append(
                ClientFlops(
                    task_id=task_id,
                    task_dir=task_dir,
                    client_id=cid,
                    client_dir=cdir,
                    net_info_path=info_path,
                    flops_m=float(flops_m),
                )
            )
    return records, warnings


def summarize(records: List[ClientFlops]) -> Dict[int, Dict[str, float]]:
    by_task: Dict[int, List[float]] = {}
    for r in records:
        by_task.setdefault(r.task_id, []).append(r.flops_m)

    out: Dict[int, Dict[str, float]] = {}
    for task_id, vals in sorted(by_task.items()):
        out[task_id] = {
            "num_clients": len(vals),
            "flops_mean_m": fmean(vals),
            "flops_max_m": max(vals),
            "flops_min_m": min(vals),
        }
    return out


def write_outputs(records: List[ClientFlops], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "task_client_flops.csv"
    summary_csv = out_dir / "task_flops_summary.csv"
    summary_json = out_dir / "task_flops_summary.json"

    # Detail CSV
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "client_id", "flops_m", "net_info_path", "task_dir", "client_dir"])
        for r in sorted(records, key=lambda x: (x.task_id, x.client_id)):
            w.writerow(
                [
                    r.task_id,
                    r.client_id,
                    f"{r.flops_m:.6f}",
                    str(r.net_info_path),
                    str(r.task_dir),
                    str(r.client_dir),
                ]
            )

    # Summary
    task_stats = summarize(records)
    all_vals = [r.flops_m for r in records]
    global_stats = {
        "num_points": len(all_vals),
        "flops_global_mean_m": fmean(all_vals) if all_vals else None,
        "flops_global_max_m": max(all_vals) if all_vals else None,
        "flops_global_min_m": min(all_vals) if all_vals else None,
    }

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "num_clients", "flops_mean_m", "flops_max_m", "flops_min_m"])
        for task_id in sorted(task_stats):
            s = task_stats[task_id]
            w.writerow(
                [
                    task_id,
                    int(s["num_clients"]),
                    f"{s['flops_mean_m']:.6f}",
                    f"{s['flops_max_m']:.6f}",
                    f"{s['flops_min_m']:.6f}",
                ]
            )
        w.writerow(
            [
                "GLOBAL",
                global_stats["num_points"],
                f"{global_stats['flops_global_mean_m']:.6f}" if global_stats["flops_global_mean_m"] is not None else "",
                f"{global_stats['flops_global_max_m']:.6f}" if global_stats["flops_global_max_m"] is not None else "",
                f"{global_stats['flops_global_min_m']:.6f}" if global_stats["flops_global_min_m"] is not None else "",
            ]
        )

    payload = {
        "task_summary": {
            str(t): {
                "num_clients": int(v["num_clients"]),
                "flops_mean_m": float(v["flops_mean_m"]),
                "flops_max_m": float(v["flops_max_m"]),
                "flops_min_m": float(v["flops_min_m"]),
            }
            for t, v in sorted(task_stats.items())
        },
        "global_summary": global_stats,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] detail:  {detail_csv}")
    print(f"[OK] summary: {summary_csv}")
    print(f"[OK] json:    {summary_json}")


def print_console_summary(records: List[ClientFlops], warnings: List[str]) -> None:
    if warnings:
        for w in warnings:
            print(w)
    if not records:
        print("[INFO] no valid flops records found")
        return

    task_stats = summarize(records)
    print("\nPer-task FLOPs summary (M):")
    print("task_id | clients | mean     | max      | min")
    for task_id in sorted(task_stats):
        s = task_stats[task_id]
        print(
            f"{task_id:>7} | {int(s['num_clients']):>7} | "
            f"{s['flops_mean_m']:>8.4f} | {s['flops_max_m']:>8.4f} | {s['flops_min_m']:>8.4f}"
        )

    vals = [r.flops_m for r in records]
    print("\nGlobal summary (M):")
    print(f"count={len(vals)}, mean={fmean(vals):.4f}, max={max(vals):.4f}, min={min(vals):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect per-task per-client FLOPs from net_info.txt and summarize mean/max/min."
    )
    parser.add_argument(
        "--exp-root",
        type=str,
        required=True,
        help="Experiment root directory containing *-taskN folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: <exp_root>/flops_summary",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (exp_root / "flops_summary")

    records, warnings = collect_all(exp_root)
    print_console_summary(records, warnings)
    write_outputs(records, out_dir)


if __name__ == "__main__":
    main()
