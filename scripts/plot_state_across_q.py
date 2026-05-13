"""
Generate IV_R4 state_across_q plot tanpa memory panel.

Skrip ini hanya membaca CSV closed_loop_data dari folder eksperimen secara read-only,
lalu menulis PNG ke thesis/src/resources/. Tidak menyentuh kode eksperimen.

Usage:
    python3 plot_state_across_q.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

THESIS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = THESIS_ROOT.parent
CSV_PATH = (
    PROJECT_ROOT
    / "batch-control/experiments/closed_loop/runs/thesis_ryzen_full__closed_loop_20260505_223558/results/closed_loop_data.csv"
)
OUT_PATH = THESIS_ROOT / "src/resources/IV_R4__state_across_q_full.png"

Q_PRESETS = ["Q1", "Q2", "Q4"]
BASELINES = [
    ("static", "Static", "tab:gray"),
    ("rule_based", "Rule-Based", "tab:blue"),
    ("pid", "PID", "tab:orange"),
]
LQR_MODES = {"Q1": "lqr_q1", "Q2": "lqr_q2", "Q4": "lqr_q4"}
LQR_COLOR = "tab:green"

PANELS = [
    ("avg_latency_ms", "Latensi rata-rata (ms)"),
    ("cpu_util", "Utilisasi CPU (%)"),
    ("io_write_ops", "Operasi I/O tulis (ops/s)"),
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 4.5))
    fig.suptitle(
        "Adaptasi variabel sistem terhadap preset Q "
        "(garis solid = LQR adaptif, garis putus-putus = baseline tidak Q-aware)",
        fontsize=11,
    )

    for ax, (var, label) in zip(axes, PANELS):
        for mode, name, color in BASELINES:
            value = df.loc[df["controller_mode"] == mode, var].mean()
            ax.axhline(value, color=color, linestyle="--", linewidth=1.5, label=name)
            ax.scatter(Q_PRESETS, [value] * len(Q_PRESETS), color=color, s=30, zorder=3)

        lqr_values = [
            df.loc[df["controller_mode"] == LQR_MODES[q], var].mean()
            for q in Q_PRESETS
        ]
        ax.plot(
            Q_PRESETS,
            lqr_values,
            color=LQR_COLOR,
            marker="o",
            linewidth=2.0,
            markersize=8,
            label="LQR",
        )
        for q, v in zip(Q_PRESETS, lqr_values):
            ax.annotate(
                f"{v:.0f}",
                (q, v),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

        ax.set_title(label)
        ax.set_xlabel("Preset Q")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
