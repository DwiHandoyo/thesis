"""
Generate IV_R9 cross-experiment comparison plot.

Membandingkan performa kontroler pada 4 eksperimen (2 mesin x 2 konfigurasi sumber daya).
Regret dihitung ulang tanpa ANN (kontroler ANN dibuang dari analisis tesis).

Sumber data: metrics_report.txt dari 4 folder run di batch-control. Read-only.
Output: thesis/src/resources/IV_R9__cross_experiment_regret.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

THESIS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = THESIS_ROOT.parent
RUNS_ROOT = PROJECT_ROOT / "batch-control/experiments/closed_loop/runs"
OUT_PATH = THESIS_ROOT / "src/resources/IV_R9__cross_experiment_cost.png"

EXPERIMENTS = [
    ("Mac Full", "thesis_mac_full__combined_20260503_q_bryson", "results"),
    ("Mac Half", "thesis_mac_half__combined_20260505_q_bryson_halfES", "results"),
    ("Ryzen Full", "thesis_ryzen_full__closed_loop_20260505_223558", "metrics"),
    ("Ryzen Half", "thesis_ryzen_half__closed_loop_20260505_155918", "metrics"),
]

CONTROLLERS = ["static", "rule_based", "pid", "lqr"]
CONTROLLER_LABELS = {"static": "Static", "rule_based": "Rule-Based", "pid": "PID", "lqr": "LQR"}
CONTROLLER_COLORS = {
    "static": "tab:gray",
    "rule_based": "tab:blue",
    "pid": "tab:orange",
    "lqr": "tab:green",
}
Q_PRESETS = ["Q1_backlog", "Q2_resource", "Q4_balanced"]
PATTERN_HEADERS = ["STEP", "RAMP", "IMPULSE", "PERIODIC_STEP", "STEP_LOW"]


def parse_report(path: Path) -> pd.DataFrame:
    """Extract cost J per (pattern, Q, controller) from metrics_report.txt."""
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"LOAD PATTERN: ", text)[1:]
    records = []
    for block in blocks:
        pattern_name = block.split("\n", 1)[0].strip()
        if pattern_name not in PATTERN_HEADERS:
            continue
        cost_section = re.search(r"--- Cost Function J ---(.+?)(?:={5,}|$)", block, re.DOTALL)
        if not cost_section:
            continue
        header_line = re.search(r"Metric\s+(.+)\n", block)
        cols = header_line.group(1).split() if header_line else []

        for q in Q_PRESETS:
            row_match = re.search(rf"\s+{q}\s+([0-9.\s]+)", cost_section.group(1))
            if not row_match:
                continue
            values = row_match.group(1).split()
            for col, val in zip(cols, values):
                ctrl = "lqr" if col.startswith("lqr_") else "ann" if col.startswith("ann") else col
                if ctrl == "lqr" or ctrl in CONTROLLERS:
                    records.append({
                        "pattern": pattern_name,
                        "q": q,
                        "raw_controller": col,
                        "controller": ctrl,
                        "cost_j": float(val),
                    })
    return pd.DataFrame(records)


def compute_cost_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Rata-ratakan cost J per kontroler atas 5 pola dan 3 preset Q (nilai mentah, bukan regret)."""
    df = df[df["controller"].isin(CONTROLLERS)].copy()
    # Untuk LQR yang punya 3 varian Q, pilih yang sesuai dengan Q preset
    q_to_lqr = {"Q1_backlog": "lqr_q1", "Q2_resource": "lqr_q2", "Q4_balanced": "lqr_q4"}
    mask_lqr = df["controller"] == "lqr"
    df["keep"] = ~mask_lqr | df.apply(
        lambda r: r["raw_controller"] == q_to_lqr.get(r["q"], ""), axis=1
    )
    df = df[df["keep"]].drop(columns=["keep"])
    # Rata-rata cost J atas semua sampel (pattern x Q yang relevan)
    return df.groupby("controller")["cost_j"].mean()


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    rows = []
    for label, folder, subdir in EXPERIMENTS:
        report_path = RUNS_ROOT / folder / subdir / "metrics_report.txt"
        df = parse_report(report_path)
        cost_per_ctrl = compute_cost_mean(df)
        for ctrl in CONTROLLERS:
            rows.append({
                "experiment": label,
                "controller": ctrl,
                "cost_j": cost_per_ctrl.get(ctrl, np.nan),
            })
    summary = pd.DataFrame(rows)

    # Print tabel ringkasan ke stdout untuk dipakai di tabel LaTeX
    pivot = summary.pivot(index="controller", columns="experiment", values="cost_j")
    pivot = pivot.reindex(CONTROLLERS)
    pivot.columns = [c for c in ["Mac Full", "Mac Half", "Ryzen Full", "Ryzen Half"]]
    print("\n=== Mean cost J per kontroler per eksperimen (tanpa ANN) ===")
    print(pivot.round(1).to_string())

    # 4-panel grouped bar (per experiment, own y-scale)
    experiments_order = [e[0] for e in EXPERIMENTS]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), sharey=False)
    bar_x = np.arange(len(CONTROLLERS))

    for ax, exp in zip(axes, experiments_order):
        vals = [pivot.loc[ctrl, exp] for ctrl in CONTROLLERS]
        colors = [CONTROLLER_COLORS[c] for c in CONTROLLERS]
        bars = ax.bar(bar_x, vals, color=colors, edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.set_xticks(bar_x)
        ax.set_xticklabels([CONTROLLER_LABELS[c] for c in CONTROLLERS], rotation=20)
        ax.set_title(exp)
        ax.set_ylabel("Cost J rata-rata" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)
        # Beri sedikit headroom untuk label
        ax_ymax = max(v for v in vals if not np.isnan(v))
        ax.set_ylim(0, ax_ymax * 1.15 if ax_ymax > 0 else 1)

    fig.suptitle(
        "Perbandingan cost J kontroler lintas mesin dan konfigurasi sumber daya "
        "(setiap panel skalanya independen, lebih rendah lebih baik)",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
