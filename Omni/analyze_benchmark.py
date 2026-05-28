#!/usr/bin/env python3
"""分析 benchmark 结果：汇总 metrics.json + npu_monitor.log，输出对比表格和折线图。"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ====================================================================== #
#  解析 npu_monitor.log（Ascend npu-smi 23.0.6 格式）                     #
# ====================================================================== #
def parse_npu_monitor(log_path: str) -> tuple[dict, dict]:
    """解析 npu_monitor.log，返回 (summary, timeseries)。

    npu-smi 每块 NPU 占两行:
      行1: | NPU_ID  Name | Health | Power  Temp  Hugepages |
      行2: | Chip         | Bus-Id | AICore(%)  Mem(MB)/Total  HBM(MB)/Total |
    """
    if not os.path.exists(log_path):
        return {}, {}

    ts_data = {}
    current_ts = None

    with open(log_path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 时间戳
        if line.startswith("==="):
            m = re.search(r"(\d{4}-\d{2}-\d{2} [\d:]+)", line)
            if m:
                try:
                    current_ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    current_ts = None
            i += 1
            continue

        # 行1: 匹配 "| NPU_ID  Name ..." 且含 "OK" 或 "NOK"
        m1 = re.match(r"\|\s*(\d+)\s+\S+\s+\|\s*\w+\s+\|", line)
        if m1:
            npu_id = int(m1.group(1))
            if npu_id not in ts_data:
                ts_data[npu_id] = {"timestamps": [], "hbm_gb": [], "aicore_pct": []}

            # 读下一行
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # 行2: | Chip | Bus-Id | AICore  Mem/Total  HBM/Total |
                # 例: | 0    | 0000:C1:00.0  | 0           0    / 0          38688/ 65536         |
                m2 = re.search(
                    r"\|\s*\d+\s+\|\s*[\w:.]+\s+\|\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)",
                    next_line,
                )
                if m2:
                    aicore_pct = int(m2.group(1))
                    hbm_used_mb = int(m2.group(4))
                    if current_ts is not None:
                        ts_data[npu_id]["timestamps"].append(current_ts)
                    ts_data[npu_id]["hbm_gb"].append(hbm_used_mb / 1024)
                    ts_data[npu_id]["aicore_pct"].append(aicore_pct)
                    i += 2
                    continue
            i += 1
            continue

        i += 1

    # 汇总
    summary = {}
    for npu_id, data in ts_data.items():
        summary[npu_id] = {}
        if data["hbm_gb"]:
            vals = data["hbm_gb"]
            summary[npu_id]["mem_mean_gb"] = sum(vals) / len(vals)
            summary[npu_id]["mem_max_gb"] = max(vals)
            summary[npu_id]["mem_min_gb"] = min(vals)
            summary[npu_id]["mem_samples"] = len(vals)
        if data["aicore_pct"]:
            vals = data["aicore_pct"]
            summary[npu_id]["util_mean"] = sum(vals) / len(vals)
            summary[npu_id]["util_max"] = max(vals)
            summary[npu_id]["util_samples"] = len(vals)

    return summary, ts_data


def load_metrics(metrics_path: str) -> dict:
    if not os.path.exists(metrics_path):
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def print_table(headers: list, rows: list):
    col_widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) + 2
        for i, h in enumerate(headers)
    ]
    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(c).ljust(w) for c, w in zip(row, col_widths)))


# ====================================================================== #
#  绘图                                                                    #
# ====================================================================== #
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_npu_timeseries(ts_data: dict, title: str, save_path: str):
    """合并绘制所有 NPU 的 HBM 显存和 AICore 利用率（4行×2列子图）。"""
    if not ts_data:
        print(f"  No NPU timeseries data, skipping plot.")
        return

    npu_ids = sorted(ts_data.keys())
    has_aicore = any(ts_data[n]["aicore_pct"] for n in npu_ids)
    n_cols = 2 if has_aicore else 1
    n_rows = len(npu_ids)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for row_idx, npu_id in enumerate(npu_ids):
        data = ts_data[npu_id]
        timestamps = data["timestamps"]
        hbm_gb = data["hbm_gb"]
        aicore = data["aicore_pct"]

        if not hbm_gb:
            continue

        if timestamps:
            t0 = timestamps[0]
            x_hbm = [(t - t0).total_seconds() for t in timestamps]
        else:
            x_hbm = list(range(len(hbm_gb)))

        # HBM
        ax_hbm = axes[row_idx][0]
        ax_hbm.plot(x_hbm, hbm_gb, color="#1f77b4", linewidth=1.5)
        ax_hbm.fill_between(x_hbm, hbm_gb, alpha=0.15, color="#1f77b4")
        ax_hbm.set_ylabel("HBM (GB)")
        mean_gb = sum(hbm_gb) / len(hbm_gb)
        max_gb = max(hbm_gb)
        ax_hbm.axhline(y=mean_gb, color="#d62728", linestyle="--", alpha=0.6, label=f"mean={mean_gb:.1f}")
        ax_hbm.axhline(y=max_gb, color="#ff7f0e", linestyle=":", alpha=0.6, label=f"max={max_gb:.1f}")
        ax_hbm.set_title(f"NPU {npu_id} — HBM", fontsize=11)
        ax_hbm.legend(loc="upper right", fontsize=8)
        ax_hbm.grid(True, alpha=0.3)

        # AICore
        if n_cols == 2:
            ax_util = axes[row_idx][1]
            if aicore:
                if timestamps:
                    x_util = [(t - t0).total_seconds() for t in timestamps[:len(aicore)]]
                else:
                    x_util = list(range(len(aicore)))
                ax_util.plot(x_util, aicore, color="#2ca02c", linewidth=1.5)
                ax_util.fill_between(x_util, aicore, alpha=0.15, color="#2ca02c")
                mean_util = sum(aicore) / len(aicore)
                max_util = max(aicore)
                ax_util.axhline(y=mean_util, color="#d62728", linestyle="--", alpha=0.6, label=f"mean={mean_util:.0f}%")
                ax_util.axhline(y=max_util, color="#ff7f0e", linestyle=":", alpha=0.6, label=f"max={max_util:.0f}%")
                ax_util.legend(loc="upper right", fontsize=8)
            ax_util.set_ylabel("AICore (%)")
            ax_util.set_title(f"NPU {npu_id} — AICore", fontsize=11)
            ax_util.set_ylim(0, 105)
            ax_util.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Time (seconds)")
    if n_cols == 2:
        axes[-1][1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_comparison(all_results: list, save_dir: str):
    concurrencies = [r["concurrency"] for r in all_results]
    qps_vals = [r.get("throughput_qps", 0) for r in all_results]
    lat_mean = [r.get("latency_mean", 0) for r in all_results]
    lat_p50 = [r.get("latency_p50", 0) or r.get("latency_median", 0) for r in all_results]
    lat_p99 = [r.get("latency_p99", 0) for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar([str(c) for c in concurrencies], qps_vals, color="#1f77b4", alpha=0.8)
    for bar, val in zip(bars, qps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Concurrency")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    x = range(len(concurrencies))
    width = 0.25
    ax.bar([i - width for i in x], lat_mean, width, label="Mean", color="#1f77b4", alpha=0.8)
    ax.bar(list(x), lat_p50, width, label="P50", color="#2ca02c", alpha=0.8)
    ax.bar([i + width for i in x], lat_p99, width, label="P99", color="#d62728", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.set_xlabel("Max Concurrency")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency vs Concurrency")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")


# ====================================================================== #
#  主函数                                                                   #
# ====================================================================== #
def main():
    parser = argparse.ArgumentParser(description="分析 benchmark 结果")
    parser.add_argument("result_dir", help="benchmark 结果目录")
    parser.add_argument("--csv", type=str, default=None, help="输出 CSV 文件路径")
    parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"ERROR: {result_dir} not found")
        sys.exit(1)

    concurrency_dirs = sorted(
        [d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith("concurrency_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not concurrency_dirs:
        print(f"ERROR: No concurrency_* directories found in {result_dir}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"  Benchmark Analysis: {result_dir.name}")
    print(f"{'='*70}\n")

    all_results = []
    for cdir in concurrency_dirs:
        concurrency = int(cdir.name.split("_")[1])
        metrics = load_metrics(cdir / "metrics.json")
        npu_summary, npu_timeseries = parse_npu_monitor(cdir / "npu_monitor.log")
        result = {"concurrency": concurrency}
        result.update(metrics)
        result["npu_summary"] = npu_summary
        result["npu_timeseries"] = npu_timeseries
        result["run_dir"] = str(cdir)
        all_results.append(result)

    # 1) 性能指标
    print("1. Performance Metrics")
    print("-" * 70)
    headers = ["Concurrency", "QPS", "Lat Mean(s)", "Lat P50(s)", "Lat P95(s)", "Lat P99(s)", "Success"]
    rows = []
    for r in all_results:
        c = r["concurrency"]
        qps = r.get("throughput_qps", 0)
        mean = r.get("latency_mean", 0)
        p50 = r.get("latency_p50", 0) or r.get("latency_median", 0)
        p95 = r.get("latency_p95", 0)
        p99 = r.get("latency_p99", 0)
        ok = r.get("completed_requests", 0)
        rows.append([c, f"{qps:.2f}", f"{mean:.1f}", f"{p50:.1f}", f"{p95:.1f}", f"{p99:.1f}", str(ok)])
    print_table(headers, rows)

    # 2) 吞吐量
    print(f"\n2. Throughput Scaling")
    print("-" * 70)
    if all_results:
        base_qps = all_results[0].get("throughput_qps", 1) or 1
        for r in all_results:
            c = r["concurrency"]
            qps = r.get("throughput_qps", 0)
            scale = qps / base_qps if base_qps else 0
            bar = "#" * int(scale * 20)
            print(f"  c={c:<4}  QPS={qps:.2f}  scale={scale:.2f}x  {bar}")

    # 3) NPU HBM
    print(f"\n3. NPU HBM Memory Usage (GB)")
    print("-" * 70)
    for r in all_results:
        c = r["concurrency"]
        npu_stats = r.get("npu_summary", {})
        if not npu_stats:
            print(f"  c={c:<4}  (no NPU monitor data)")
            continue
        parts = []
        for npu_id in sorted(npu_stats.keys()):
            s = npu_stats[npu_id]
            mean_gb = s.get("mem_mean_gb", 0)
            max_gb = s.get("mem_max_gb", 0)
            parts.append(f"NPU{npu_id}: mean={mean_gb:.1f} max={max_gb:.1f}GB")
        print(f"  c={c:<4}  {' | '.join(parts)}")

    # 4) NPU AICore
    print(f"\n4. NPU AICore Utilization (%)")
    print("-" * 70)
    for r in all_results:
        c = r["concurrency"]
        npu_stats = r.get("npu_summary", {})
        if not npu_stats:
            print(f"  c={c:<4}  (no NPU monitor data)")
            continue
        parts = []
        for npu_id in sorted(npu_stats.keys()):
            s = npu_stats[npu_id]
            mean_util = s.get("util_mean", 0)
            max_util = s.get("util_max", 0)
            parts.append(f"NPU{npu_id}: mean={mean_util:.0f}% max={max_util:.0f}%")
        print(f"  c={c:<4}  {' | '.join(parts)}")

    # 5) 阶段耗时
    print(f"\n5. Stage Durations (s)")
    print("-" * 70)
    for r in all_results:
        c = r["concurrency"]
        stages_mean = r.get("stage_durations_mean", {})
        stages_p50 = r.get("stage_durations_p50", {})
        if not stages_mean:
            print(f"  c={c:<4}  (no stage data)")
            continue
        parts = []
        for stage in sorted(stages_mean.keys()):
            parts.append(f"{stage}: mean={stages_mean[stage]:.2f} p50={stages_p50.get(stage, 0):.2f}")
        print(f"  c={c:<4}  {' | '.join(parts)}")

    # 6) 图表
    if not args.no_plot:
        print(f"\n6. Generating Plots")
        print("-" * 70)
        for r in all_results:
            c = r["concurrency"]
            ts = r.get("npu_timeseries", {})
            if ts:
                plot_npu_timeseries(
                    ts,
                    title=f"NPU Status — Concurrency={c}",
                    save_path=os.path.join(r["run_dir"], "npu_timeseries.png"),
                )
        plot_comparison(all_results, str(result_dir))

    # 7) CSV
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["concurrency", "qps", "latency_mean", "latency_p50",
                             "latency_p95", "latency_p99", "success"])
            for r in all_results:
                writer.writerow([
                    r["concurrency"],
                    f"{r.get('throughput_qps', 0):.2f}",
                    f"{r.get('latency_mean', 0):.1f}",
                    f"{r.get('latency_p50', 0) or r.get('latency_median', 0):.1f}",
                    f"{r.get('latency_p95', 0):.1f}",
                    f"{r.get('latency_p99', 0):.1f}",
                    r.get("completed_requests", 0),
                ])
        print(f"\nCSV saved to {args.csv}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
