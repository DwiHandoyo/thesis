"""
Generate IV_R10 knee-point plot dari data capacity benchmark.

Plot menampilkan throughput vs batch_size dengan CPU sebagai sumbu sekunder, dan
menandai knee point di batch=500 secara visual.

Sumber data: capacity_benchmark_*.json (read-only).
Output: thesis/src/resources/IV_R10__knee_point.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

THESIS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = THESIS_ROOT.parent
BENCHMARK_JSON = (
    PROJECT_ROOT
    / "batch-control/experiments/open_loop/results/capacity_benchmark/20260414_222623/capacity_benchmark_20260414_222623.json"
)
OUT_PATH = THESIS_ROOT / "src/resources/IV_R10__knee_point.png"

KNEE_BATCH = 500


def main() -> None:
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)
    results = data["benchmark_results"]

    batch = [r["batch_size"] for r in results]
    throughput = [r["throughput_msg_per_s"] for r in results]
    cpu = [r["avg_cpu"] for r in results]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    # Throughput axis (primary, kiri)
    color_tput = "tab:blue"
    ax1.plot(
        batch,
        throughput,
        marker="o",
        markersize=8,
        linewidth=2,
        color=color_tput,
        label="Throughput",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Batch size (skala log)")
    ax1.set_ylabel("Throughput (msg/s)", color=color_tput)
    ax1.tick_params(axis="y", labelcolor=color_tput)
    ax1.set_xticks(batch)
    ax1.set_xticklabels([str(b) for b in batch])
    ax1.grid(True, which="both", alpha=0.3)

    # Annotate throughput values
    for b, t in zip(batch, throughput):
        ax1.annotate(
            f"{t:,.0f}".replace(",", "."),
            (b, t),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color=color_tput,
        )

    # CPU axis (sekunder, kanan)
    ax2 = ax1.twinx()
    color_cpu = "tab:red"
    ax2.plot(
        batch,
        cpu,
        marker="s",
        markersize=7,
        linewidth=1.5,
        linestyle="--",
        color=color_cpu,
        label="CPU",
        alpha=0.85,
    )
    ax2.set_ylabel("Utilisasi CPU (%)", color=color_cpu)
    ax2.tick_params(axis="y", labelcolor=color_cpu)
    ax2.set_ylim(0, max(cpu) * 1.15)

    for b, c in zip(batch, cpu):
        ax2.annotate(
            f"{c:.1f}\\%".replace("\\", ""),
            (b, c),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            fontsize=8,
            color=color_cpu,
        )

    # Knee point marker
    knee_idx = batch.index(KNEE_BATCH)
    knee_tput = throughput[knee_idx]
    knee_cpu = cpu[knee_idx]
    ax1.axvline(KNEE_BATCH, color="black", linestyle=":", linewidth=1.2, alpha=0.6)
    ax1.annotate(
        f"Knee point\nbatch = {KNEE_BATCH}\nCPU = {knee_cpu:.1f}\\%".replace("\\", ""),
        xy=(KNEE_BATCH, knee_tput),
        xytext=(40, knee_tput * 0.95),
        fontsize=10,
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray"),
    )

    # Legend gabungan
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.suptitle(
        "Identifikasi Knee Point dari Capacity Benchmark "
        "(throughput jenuh setelah batch=500 sementara CPU melonjak)",
        fontsize=10.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
