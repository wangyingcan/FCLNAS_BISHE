#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


TASK_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)-task(?P<task>\d+)$")
TIMING_RE = re.compile(
    r"(search|retrain)_timing\s+local_compute\s+([0-9.]+)m,\s+aggregation\s+([0-9.]+)m,\s+"
    r"evaluation\s+([0-9.]+)m,\s+checkpoint_io\s+([0-9.]+)m,\s+algorithm\s+([0-9.]+)m,\s+wall\s+([0-9.]+)m"
)
FORGETTING_RE = re.compile(r"forgetting_F_max:\s*([-+]?[0-9]*\.?[0-9]+)")


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def load_jsonl(path: str) -> List[dict]:
    records = []
    if not os.path.isfile(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def discover_task_dirs(exp_prefix: str) -> List[Tuple[int, str]]:
    task_dirs = []
    # Case 1: user passes prefix without task suffix.
    for path in glob.glob(f"{exp_prefix}-task*"):
        m = TASK_SUFFIX_RE.match(path)
        if m is None:
            continue
        task_dirs.append((int(m.group("task")), path))
    # Case 2: user passes a single task path directly.
    if not task_dirs and os.path.isdir(exp_prefix):
        m = TASK_SUFFIX_RE.match(exp_prefix)
        if m is not None:
            task_dirs.append((int(m.group("task")), exp_prefix))
    task_dirs.sort(key=lambda x: x[0])
    return task_dirs


def pick_last_retrain_test_summary(stats_records: List[dict]) -> Optional[dict]:
    if not stats_records:
        return None
    filtered = [r for r in stats_records if r.get("phase") == "retrain_test_summary"]
    if not filtered:
        return None
    learned = [r for r in filtered if r.get("tag") == "learned_net"]
    target = learned if learned else filtered
    target.sort(key=lambda r: int(r.get("round", -1)))
    return target[-1]


def extract_final_per_client_current_task(
    per_client_records: List[dict],
) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    # Returns: final_round, client_current_task_top1, client_test_top1
    test_rows = []
    for r in per_client_records:
        if r.get("phase") != "retrain":
            continue
        if "test_top1" not in r:
            continue
        rd = int(r.get("round", -1))
        if rd < 0:
            continue
        test_rows.append(r)
    if not test_rows:
        return -1, {}, {}
    final_round = max(int(r.get("round", -1)) for r in test_rows)
    cur_map: Dict[int, float] = {}
    test_map: Dict[int, float] = {}
    for r in test_rows:
        if int(r.get("round", -1)) != final_round:
            continue
        cid = int(r.get("client_id", -1))
        if cid < 0:
            continue
        cur_top1 = safe_float(r.get("current_task_top1"))
        test_top1 = safe_float(r.get("test_top1"))
        if cur_top1 is not None:
            cur_map[cid] = cur_top1
        if test_top1 is not None:
            test_map[cid] = test_top1
    return final_round, cur_map, test_map


def parse_timing_log(path: str) -> Dict[str, float]:
    out = {
        "timing_rounds": 0,
        "local_compute_min": 0.0,
        "aggregation_min": 0.0,
        "evaluation_min": 0.0,
        "checkpoint_io_min": 0.0,
        "algorithm_min": 0.0,
        "wall_min": 0.0,
    }
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TIMING_RE.search(line)
            if m is None:
                continue
            out["timing_rounds"] += 1
            out["local_compute_min"] += float(m.group(2))
            out["aggregation_min"] += float(m.group(3))
            out["evaluation_min"] += float(m.group(4))
            out["checkpoint_io_min"] += float(m.group(5))
            out["algorithm_min"] += float(m.group(6))
            out["wall_min"] += float(m.group(7))
    return out


def parse_last_forgetting_fmax(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    last_val = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = FORGETTING_RE.search(line)
            if m is not None:
                last_val = safe_float(m.group(1))
    return last_val


def summarize_experiment(exp_prefix: str) -> Tuple[dict, List[dict], List[dict]]:
    task_dirs = discover_task_dirs(exp_prefix)
    if not task_dirs:
        raise FileNotFoundError(f"未找到任务目录: {exp_prefix}-task*")

    task_rows: List[dict] = []
    # client_task_scores[(client_id, task_id)] = score
    client_task_scores: Dict[Tuple[int, int], float] = {}
    # cost totals
    search_cost = defaultdict(float)
    retrain_cost = defaultdict(float)
    final_forgetting_fmax = None

    for task_id, task_dir in task_dirs:
        logs_dir = os.path.join(task_dir, "logs")
        stats_path = os.path.join(logs_dir, "client_test_top1_stats.jsonl")
        metric_path = os.path.join(logs_dir, "per_client_metrics.jsonl")
        test_log_path = os.path.join(logs_dir, "test.log")

        stats_records = load_jsonl(stats_path)
        summary = pick_last_retrain_test_summary(stats_records)

        per_client_records = load_jsonl(metric_path)
        final_round, cur_map, test_map = extract_final_per_client_current_task(per_client_records)
        used_map = cur_map if cur_map else test_map
        for cid, score in used_map.items():
            client_task_scores[(cid, task_id)] = float(score)

        task_row = {
            "exp_prefix": exp_prefix,
            "task_id": task_id,
            "task_dir": task_dir,
            "final_round_from_per_client": final_round,
            "available_clients_in_task": len(used_map),
            "current_task_top1_mean": None,
            "current_task_top1_weighted_mean": None,
            "current_task_top1_std": None,
            "current_task_top1_min": None,
            "current_task_top1_max": None,
            "test_top1_mean": None,
            "test_top1_weighted_mean": None,
            "test_top1_std": None,
            "test_top1_min": None,
            "test_top1_max": None,
            "test_loss_mean": None,
            "test_top5_mean": None,
            "final_forgetting_fmax": None,
        }
        if summary is not None:
            for key in [
                "current_task_top1_mean",
                "current_task_top1_weighted_mean",
                "current_task_top1_std",
                "current_task_top1_min",
                "current_task_top1_max",
                "test_top1_mean",
                "test_top1_weighted_mean",
                "test_top1_std",
                "test_top1_min",
                "test_top1_max",
                "test_loss_mean",
                "test_top5_mean",
            ]:
                task_row[key] = safe_float(summary.get(key))
        forgetting_val = parse_last_forgetting_fmax(test_log_path)
        task_row["final_forgetting_fmax"] = forgetting_val
        if forgetting_val is not None:
            final_forgetting_fmax = forgetting_val
        task_rows.append(task_row)

        search_timing = parse_timing_log(os.path.join(logs_dir, "search.log"))
        retrain_timing = parse_timing_log(os.path.join(logs_dir, "retrain.log"))
        for k, v in search_timing.items():
            search_cost[k] += v
        for k, v in retrain_timing.items():
            retrain_cost[k] += v

    # Build client means across tasks for personalized metrics
    client_to_scores: Dict[int, List[float]] = defaultdict(list)
    for (cid, _task), score in client_task_scores.items():
        client_to_scores[cid].append(score)
    client_rows: List[dict] = []
    for cid in sorted(client_to_scores.keys()):
        vals = client_to_scores[cid]
        client_rows.append(
            {
                "exp_prefix": exp_prefix,
                "client_id": cid,
                "task_count": len(vals),
                "client_task_mean_acc": mean(vals),
            }
        )

    client_means = [row["client_task_mean_acc"] for row in client_rows]
    if client_means:
        final_avgacc = mean(client_means)
        worst_client = min(client_means)
        best_client = max(client_means)
        client_std = pstdev(client_means) if len(client_means) > 1 else 0.0
    else:
        final_avgacc = None
        worst_client = None
        best_client = None
        client_std = None

    # Task-level mean of current-task accuracy
    task_mean_vals = [
        r["current_task_top1_mean"] if r["current_task_top1_mean"] is not None else r["test_top1_mean"]
        for r in task_rows
        if (r["current_task_top1_mean"] is not None or r["test_top1_mean"] is not None)
    ]
    task_avgacc_mean = mean(task_mean_vals) if task_mean_vals else None

    summary_row = {
        "exp_prefix": exp_prefix,
        "task_count": len(task_rows),
        "client_count": len(client_rows),
        "Final_AvgAcc": final_avgacc,
        "Worst_client_Acc": worst_client,
        "Best_client_Acc": best_client,
        "Client_Std": client_std,
        "Task_AvgAcc_Mean": task_avgacc_mean,
        "Final_Forgetting_Fmax": final_forgetting_fmax,
        "Search_algorithm_min_total": search_cost["algorithm_min"],
        "Search_wall_min_total": search_cost["wall_min"],
        "Retrain_algorithm_min_total": retrain_cost["algorithm_min"],
        "Retrain_wall_min_total": retrain_cost["wall_min"],
        "Search_timing_rounds": int(search_cost["timing_rounds"]),
        "Retrain_timing_rounds": int(retrain_cost["timing_rounds"]),
    }
    return summary_row, task_rows, client_rows


def write_csv(path: str, rows: List[dict], preferred_cols: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = set()
    for r in rows:
        keys.update(r.keys())
    cols = []
    if preferred_cols:
        for c in preferred_cols:
            if c in keys:
                cols.append(c)
                keys.remove(c)
    cols.extend(sorted(keys))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    parser = argparse.ArgumentParser(description="导出 FCL 实验评估指标（性能与耗时）")
    parser.add_argument(
        "--exp_prefixes",
        nargs="+",
        required=True,
        help="实验前缀列表，例如 ./outputs/phase3/p3_grad_noniid_priorflex_s0",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./metrics_exports",
        help="导出目录",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="fcl_metrics",
        help="导出文件名前缀",
    )
    args = parser.parse_args()

    summary_rows: List[dict] = []
    task_rows_all: List[dict] = []
    client_rows_all: List[dict] = []

    for exp_prefix in args.exp_prefixes:
        summary_row, task_rows, client_rows = summarize_experiment(exp_prefix)
        summary_rows.append(summary_row)
        task_rows_all.extend(task_rows)
        client_rows_all.extend(client_rows)

    summary_path = os.path.join(args.out_dir, f"{args.out_prefix}_summary.csv")
    task_path = os.path.join(args.out_dir, f"{args.out_prefix}_task_metrics.csv")
    client_path = os.path.join(args.out_dir, f"{args.out_prefix}_client_metrics.csv")

    write_csv(
        summary_path,
        summary_rows,
        preferred_cols=[
            "exp_prefix",
            "task_count",
            "client_count",
            "Final_AvgAcc",
            "Worst_client_Acc",
            "Best_client_Acc",
            "Client_Std",
            "Task_AvgAcc_Mean",
            "Final_Forgetting_Fmax",
            "Search_algorithm_min_total",
            "Search_wall_min_total",
            "Retrain_algorithm_min_total",
            "Retrain_wall_min_total",
            "Search_timing_rounds",
            "Retrain_timing_rounds",
        ],
    )
    write_csv(task_path, task_rows_all)
    write_csv(
        client_path,
        client_rows_all,
        preferred_cols=["exp_prefix", "client_id", "task_count", "client_task_mean_acc"],
    )

    print("导出完成:")
    print(f"  summary: {os.path.abspath(summary_path)}")
    print(f"  task   : {os.path.abspath(task_path)}")
    print(f"  client : {os.path.abspath(client_path)}")


if __name__ == "__main__":
    main()
