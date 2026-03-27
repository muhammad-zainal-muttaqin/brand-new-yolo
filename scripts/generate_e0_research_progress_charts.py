from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "outputs" / "reports" / "run_ledger.csv"
OUT_DIR = ROOT / "outputs" / "figures"


PHASE_COLORS = {
    "phase0": "#64748B",
    "phase1": "#8B5CF6",
    "phase2": "#0EA5E9",
    "phase3": "#F97316",
}

METRICS = [
    ("map50", "mAP50", "#1D4ED8"),
    ("map50_95", "mAP50-95", "#059669"),
    ("precision", "Precision", "#D97706"),
    ("recall", "Recall", "#DC2626"),
]


def load_detection_progress() -> pd.DataFrame:
    df = pd.read_csv(LEDGER_PATH)
    df = df[df["status"] == "completed"].copy()

    exclude_mask = (
        df["run_name"].str.contains(r"p1a_stage1_singlecls", regex=True, na=False)
        | df["run_name"].str.contains(r"p1a_stage2_cls", regex=True, na=False)
        | df["model"].fillna("").str.contains(r"-cls", regex=False)
    )
    df = df[~exclude_mask].copy()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df["exp_idx"] = np.arange(1, len(df) + 1)
    return df


def draw_phase_bands(ax: plt.Axes, df: pd.DataFrame, y_top: float) -> None:
    ranges = (
        df.groupby("phase", sort=False)["exp_idx"]
        .agg(["min", "max"])
        .reset_index()
    )

    for _, row in ranges.iterrows():
        xmin = row["min"] - 0.5
        xmax = row["max"] + 0.5
        ax.axvspan(xmin, xmax, color=PHASE_COLORS[row["phase"]], alpha=0.05, lw=0)
        ax.text(
            (xmin + xmax) / 2,
            y_top,
            row["phase"].upper(),
            ha="center",
            va="bottom",
            fontsize=12,
            color=PHASE_COLORS[row["phase"]],
            fontweight="bold",
        )

    for boundary in ranges["max"].tolist()[:-1]:
        ax.axvline(boundary + 0.5, color="#CBD5E1", lw=1.2, ls="--", zorder=0)


def make_chart(df: pd.DataFrame, metric_key: str, metric_label: str, accent: str) -> None:
    series = df[metric_key].astype(float)
    running_best = series.cummax()
    is_new_best = series.eq(running_best) & series.ne(running_best.shift(fill_value=-np.inf))

    ymin = float(series.min())
    ymax = float(series.max())
    ypad = max((ymax - ymin) * 0.12, 0.01)

    fig, ax = plt.subplots(figsize=(15, 7.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_phase_bands(ax, df, ymax + ypad * 0.68)

    ax.plot(
        df["exp_idx"],
        series,
        color="#94A3B8",
        lw=1.6,
        alpha=0.75,
        zorder=1,
    )

    ax.step(
        df["exp_idx"],
        running_best,
        where="post",
        color=accent,
        lw=2.8,
        alpha=0.95,
        zorder=2,
    )

    for phase, phase_df in df.groupby("phase", sort=False):
        ax.scatter(
            phase_df["exp_idx"],
            phase_df[metric_key],
            s=64,
            color=PHASE_COLORS[phase],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    best_points = df[is_new_best]
    ax.scatter(
        best_points["exp_idx"],
        best_points[metric_key],
        s=110,
        facecolor="white",
        edgecolor=accent,
        linewidth=2.2,
        zorder=4,
    )

    for _, row in best_points.iterrows():
        ax.text(
            row["exp_idx"],
            row[metric_key] + ypad * 0.1,
            f"{row[metric_key]:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=accent,
            fontweight="bold",
        )

    ax.set_xlim(0.5, df["exp_idx"].max() + 0.5)
    ax.set_ylim(ymin - ypad, ymax + ypad * 0.95)
    ax.set_title(metric_label, fontsize=24, fontweight="bold", pad=18)
    ax.set_xlabel("Urutan eksperimen", fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.grid(axis="y", color="#E2E8F0", lw=1.0)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")

    step = 5 if len(df) <= 80 else 10
    xticks = list(range(1, int(df["exp_idx"].max()) + 1, step))
    if xticks[-1] != int(df["exp_idx"].max()):
        xticks.append(int(df["exp_idx"].max()))
    ax.set_xticks(xticks)

    fig.tight_layout()
    out_path = OUT_DIR / f"e0_research_progress_{metric_key}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_detection_progress()
    for metric_key, metric_label, accent in METRICS:
        make_chart(df, metric_key, metric_label, accent)
        print(OUT_DIR / f"e0_research_progress_{metric_key}.png")


if __name__ == "__main__":
    main()
