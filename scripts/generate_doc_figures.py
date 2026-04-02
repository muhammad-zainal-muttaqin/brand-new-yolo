"""Generate documentation figures for Brand-New-YOLO research docs.

Reads CSV/JSON artefacts and creates publication-quality PNG charts.

Usage:
    python scripts/generate_doc_figures.py          # generate all
    python scripts/generate_doc_figures.py --phase 0 1 2 3   # selective
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "outputs" / "phase0"
P1 = ROOT / "outputs" / "phase1"
P2 = ROOT / "outputs" / "phase2"
P3 = ROOT / "outputs" / "phase3"
RUNS = ROOT / "runs" / "detect" / "runs" / "e0"

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
DPI = 180

CLASS_COLORS = {
    "B1": "#EF4444",
    "B2": "#F97316",
    "B3": "#6366F1",
    "B4": "#10B981",
}

METRIC_COLORS = {
    "precision": "#D97706",
    "recall":    "#DC2626",
    "mAP50":     "#1D4ED8",
    "mAP50-95":  "#059669",
    "b4_recall":  "#10B981",
}

ERROR_COLORS = {
    "false_positive":    "#EF4444",
    "B2_B3_confusion":   "#F97316",
    "B4_missed":         "#6366F1",
    "B3_B4_confusion":   "#10B981",
}


def _style_ax(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(labelsize=11)
    ax.set_facecolor("white")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  [OK] {path.relative_to(ROOT)}")


def _phase3_csv(name: str) -> pd.DataFrame:
    path = P3 / name
    if not path.exists():
        print(f"  [SKIP] missing {path.relative_to(ROOT)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def _candidate_color(candidate: str) -> str:
    palette = {
        "yolo11m": "#1D4ED8",
        "yolov8s": "#F97316",
        "two_stage": "#10B981",
        "gtcrop": "#059669",
    }
    return palette.get(candidate, "#64748B")


def _metric_color(metric: str) -> str:
    aliases = {
        "map50": "mAP50",
        "map50_95": "mAP50-95",
    }
    return METRIC_COLORS.get(metric, METRIC_COLORS.get(aliases.get(metric, ""), "#64748B"))


def _canonical_class_order(columns: list[str]) -> list[str]:
    return [name for name in ["B1", "B2", "B3", "B4"] if name in columns]


def _ordered_candidates(values: list[str]) -> list[str]:
    order = {"yolo11m": 0, "yolov8s": 1}
    return sorted(values, key=lambda value: (order.get(value, 99), value))


def _phase3_one_stage_metrics(split: str = "test", checkpoint: str = "last") -> pd.DataFrame:
    df = _phase3_csv("final_metrics.csv")
    if df.empty:
        return df
    required = {"branch", "checkpoint", "split"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df[
        (df["branch"] == "one_stage")
        & (df["checkpoint"] == checkpoint)
        & (df["split"] == split)
    ].copy()


def _phase3_pick_row(
    df: pd.DataFrame,
    *,
    branch: str,
    checkpoint: str,
    split: str,
    candidate: str | None = None,
) -> pd.Series | None:
    if df.empty:
        return None
    subset = df[
        (df["branch"] == branch)
        & (df["checkpoint"] == checkpoint)
        & (df["split"] == split)
    ].copy()
    if candidate is not None:
        subset = subset[subset["candidate"] == candidate]
    if subset.empty:
        return None
    return subset.iloc[0]


def _phase3_confusion_counts(
    df: pd.DataFrame,
    *,
    branch: str,
    checkpoint: str,
    split: str,
    candidate: str | None = None,
) -> tuple[list[str], pd.DataFrame] | tuple[None, None]:
    if df.empty:
        return None, None
    subset = df[
        (df["branch"] == branch)
        & (df["checkpoint"] == checkpoint)
        & (df["split"] == split)
    ].copy()
    if candidate is not None:
        subset = subset[subset["candidate"] == candidate]
    if subset.empty:
        return None, None
    class_order = _canonical_class_order(subset.columns.tolist())
    if not class_order:
        return None, None
    matrix = subset.set_index("true_class").reindex(class_order)
    return class_order, matrix


# ===================================================================
# Phase 0
# ===================================================================
def f1_class_distribution() -> None:
    df = pd.read_csv(P0 / "class_distribution.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=DPI)
    _style_ax(ax)
    colors = [CLASS_COLORS[n] for n in df["class_name"]]
    bars = ax.barh(df["class_name"], df["count"], color=colors, edgecolor="white", height=0.6)
    for bar, share in zip(bars, df["share"]):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                f"{share*100:.1f}%", va="center", fontsize=11, fontweight="bold", color="#334155")
    ax.set_xlabel("Instance Count", fontsize=12)
    ax.set_title("Distribusi Kelas dalam Dataset", fontsize=16, fontweight="bold", pad=14)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P0 / "figures" / "eda_class_distribution.png")


def f2_bbox_size() -> None:
    df = pd.read_csv(P0 / "bbox_stats.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), dpi=DPI)
    fig.suptitle("Perbandingan Ukuran Bounding Box per Kelas", fontsize=16, fontweight="bold", y=1.02)
    metrics = [
        ("median_width_px", "Median Width (px)"),
        ("median_height_px", "Median Height (px)"),
        ("median_area_norm", "Median Area (normalized)"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        _style_ax(ax)
        colors = [CLASS_COLORS[n] for n in df["class_name"]]
        bars = ax.bar(df["class_name"], df[col], color=colors, edgecolor="white", width=0.55)
        for bar in bars:
            fmt = f"{bar.get_height():.4f}" if "norm" in col else f"{bar.get_height():.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=9, fontweight="bold", color="#334155")
        ax.set_ylabel(label, fontsize=10)
        ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P0 / "figures" / "eda_bbox_size_comparison.png")


def f3_resolution() -> None:
    df = pd.read_csv(P0 / "resolution_sweep.csv")
    df["res"] = df["imgsz"].astype(str)
    metrics = ["map50", "map50_95", "precision", "recall"]
    labels = ["mAP50", "mAP50-95", "Precision", "Recall"]
    means = df.groupby("res")[metrics].mean()
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=DPI)
    _style_ax(ax)
    x = np.arange(len(labels))
    w = 0.32
    c640, c1024 = "#1D4ED8", "#F97316"
    bars1 = ax.bar(x - w/2, means.loc["640"], w, label="640", color=c640, edgecolor="white")
    bars2 = ax.bar(x + w/2, means.loc["1024"], w, label="1024", color=c1024, edgecolor="white")
    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#334155")
    for _, row in df.iterrows():
        color = c640 if row["imgsz"] == 640 else c1024
        offset = -w/2 if row["imgsz"] == 640 else w/2
        for i, m in enumerate(metrics):
            ax.scatter(i + offset, row[m], s=36, color=color, edgecolor="white",
                       linewidth=0.8, zorder=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Perbandingan Resolusi: 640 vs 1024", fontsize=16, fontweight="bold", pad=14)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P0 / "figures" / "p0_resolution_comparison.png")


def f4_learning_curve() -> None:
    df = pd.read_csv(P0 / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)
    _style_ax(ax)
    fractions = df["fraction"] * 100
    for metric, label, color in [("map50", "mAP50", "#1D4ED8"), ("map50_95", "mAP50-95", "#059669")]:
        ax.plot(fractions, df[metric], marker="o", markersize=8, lw=2.5, color=color, label=label, zorder=3)
        for xv, yv in zip(fractions, df[metric]):
            ax.text(xv, yv + 0.006, f"{yv:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)
    ax.set_xlabel("Data Fraction (%)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Learning Curve — mAP vs Fraksi Data", fontsize=16, fontweight="bold", pad=14)
    ax.set_xticks([25, 50, 75, 100])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P0 / "figures" / "p0_learning_curve.png")


# ===================================================================
# Phase 1
# ===================================================================
def f5_architecture_benchmark() -> None:
    df = pd.read_csv(P1 / "architecture_benchmark.csv").sort_values("mean_map50")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=DPI)
    _style_ax(ax)
    models = df["model"].str.replace(".pt", "", regex=False)
    y = np.arange(len(models))
    is_top3 = np.arange(len(models)) >= (len(models) - 3)
    colors = ["#1D4ED8" if t else "#94A3B8" for t in is_top3]
    ax.barh(y, df["mean_map50"], xerr=df["std_map50"], height=0.6,
            color=colors, edgecolor="white", capsize=3, error_kw={"lw": 1.2, "color": "#64748B"})
    for i, (val, std) in enumerate(zip(df["mean_map50"], df["std_map50"])):
        ax.text(val + std + 0.003, i, f"{val:.4f}", va="center", fontsize=9,
                fontweight="bold" if is_top3[i] else "normal",
                color="#1D4ED8" if is_top3[i] else "#64748B")
    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel("Mean mAP50 (2 seeds)", fontsize=12)
    ax.set_title("Architecture Benchmark — Ranking by mAP50", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P1 / "figures" / "p1_architecture_benchmark.png")


def f6_one_vs_two_stage() -> None:
    p3 = _phase3_csv("final_metrics.csv")
    required = {"branch", "candidate", "checkpoint", "split", "map50", "map50_95", "top1_acc", "weighted_f1"}
    if not p3.empty and required.issubset(p3.columns):
        one_stage = p3[
            (p3["branch"] == "one_stage")
            & (p3["checkpoint"] == "last")
            & (p3["split"] == "test")
        ].copy()
        stage1 = p3[
            (p3["branch"] == "two_stage_stage1")
            & (p3["checkpoint"] == "last")
            & (p3["split"] == "test")
        ].copy()
        gtcrop = p3[
            (p3["branch"] == "two_stage_gtcrop")
            & (p3["checkpoint"] == "last")
            & (p3["split"] == "test")
        ].copy()
        e2e = p3[
            (p3["branch"] == "two_stage_end_to_end")
            & (p3["checkpoint"] == "last")
            & (p3["split"] == "test")
        ].copy()

        if not one_stage.empty and not stage1.empty and not gtcrop.empty and not e2e.empty:
            one_stage["map50"] = pd.to_numeric(one_stage["map50"], errors="coerce").fillna(0.0)
            selected = one_stage.sort_values("map50", ascending=False).iloc[0]

            categories = [
                f"One-Stage\n{selected['candidate']}\nmAP50",
                f"One-Stage\n{selected['candidate']}\nWeighted F1",
                "Two-Stage\nStage1\nmAP50-95",
                "Two-Stage\nGT-crop\nTop-1",
                "Two-Stage\nEnd-to-End\nWeighted F1",
            ]
            values = [
                float(selected["map50"]),
                float(selected["weighted_f1"]),
                float(stage1.iloc[0]["map50_95"]),
                float(gtcrop.iloc[0]["top1_acc"]),
                float(e2e.iloc[0]["weighted_f1"]),
            ]
            colors_list = ["#1D4ED8", "#1D4ED8", "#F97316", "#10B981", "#F97316"]

            fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=DPI)
            _style_ax(ax)
            bars = ax.bar(categories, values, color=colors_list, edgecolor="white", width=0.62)
            for idx, bar in enumerate(bars):
                if idx == 3:
                    bar.set_hatch("//")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#334155",
                )
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("One-Stage vs Two-Stage Reference (Phase 3 Final)", fontsize=16, fontweight="bold", pad=14)
            ax.grid(axis="y", color="#E2E8F0", lw=0.8)
            ax.set_ylim(0, max(values) * 1.22)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor="#1D4ED8", label="One-stage end-to-end candidate utama"),
                Patch(facecolor="#F97316", label="Two-stage deployed components"),
                Patch(facecolor="#10B981", hatch="//", label="GT-crop classifier upper bound"),
            ], fontsize=10, framealpha=0.9)
            fig.text(
                0.5,
                0.01,
                "GT-crop adalah upper bound classifier; deployed result ada di end-to-end weighted F1.",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#475569",
            )
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            _save(fig, P1 / "figures" / "p1_one_vs_two_stage.png")
            return

    one = pd.read_csv(P1 / "one_stage_results.csv")
    two = pd.read_csv(P1 / "two_stage_results.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=DPI)
    _style_ax(ax)
    categories = ["One-Stage\nmAP50", "One-Stage\nmAP50-95",
                   "Two-Stage Det\nmAP50-95", "Two-Stage Cls\nTop-1 Acc"]
    values = [
        one["map50"].mean(), one["map50_95"].mean(),
        two[two["component"] == "stage1_detector_singlecls"]["metric_primary"].mean(),
        two[two["component"] == "stage2_classifier_gtcrop"]["metric_primary"].mean(),
    ]
    colors_list = ["#1D4ED8", "#1D4ED8", "#F97316", "#F97316"]
    bars = ax.bar(categories, values, color=colors_list, edgecolor="white", width=0.55)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#334155")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("One-Stage vs Two-Stage Pipeline", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.set_ylim(0, max(values) * 1.18)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#1D4ED8", label="One-Stage (4-class YOLO)"),
        Patch(facecolor="#F97316", label="Two-Stage (detect + classify)"),
    ], fontsize=10, framealpha=0.9)
    fig.tight_layout()
    _save(fig, P1 / "figures" / "p1_one_vs_two_stage.png")


def f15_per_class_heatmap() -> None:
    """Per-class mAP50 heatmap across all Phase 1B architectures."""
    df = pd.read_csv(P1 / "per_class_metrics.csv")
    # Average across seeds
    agg = df.groupby(["model", "class_name"])["map50"].mean().reset_index()
    pivot = agg.pivot(index="model", columns="class_name", values="map50")
    pivot = pivot[["B1", "B2", "B3", "B4"]]
    # Sort by overall mean
    pivot["_mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_mean", ascending=True)
    pivot = pivot.drop(columns=["_mean"])
    # Clean model names
    pivot.index = pivot.index.str.replace(".pt", "", regex=False)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=DPI)
    data = pivot.values
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(pivot.columns, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    # Annotate cells
    for i in range(len(pivot)):
        for j in range(4):
            val = data[i, j]
            color = "white" if val < 0.35 or val > 0.7 else "#1E293B"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)
    ax.set_title("Per-Class mAP50 Across Architectures (Phase 1B)", fontsize=15, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, label="mAP50", shrink=0.8, pad=0.02)
    fig.tight_layout()
    _save(fig, P1 / "figures" / "p1_per_class_heatmap.png")


# ===================================================================
# Phase 2
# ===================================================================
def f7_lr_sweep() -> None:
    df = pd.read_csv(P2 / "lr_sweep.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=DPI)
    _style_ax(ax)
    labels = df["option"].values
    x = np.arange(len(labels))
    w = 0.22
    for i, (metric, name, color) in enumerate([
        ("mean_map50", "mAP50", "#1D4ED8"),
        ("mean_map50_95", "mAP50-95", "#059669"),
        ("mean_b4_recall", "B4 Recall", "#10B981"),
    ]):
        bars = ax.bar(x + (i - 1) * w, df[metric], w, label=name, color=color, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Learning Rate Sweep", fontsize=16, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P2 / "figures" / "p2_lr_sweep.png")


def f8_batch_aug_sweep() -> None:
    batch_df = pd.read_csv(P2 / "batch_sweep.csv")
    aug_df = pd.read_csv(P2 / "aug_sweep.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=DPI)
    fig.suptitle("Batch Size & Augmentation Sweep", fontsize=16, fontweight="bold", y=1.02)
    for ax, df, title in [
        (ax1, batch_df, "Batch Size Sweep"),
        (ax2, aug_df, "Augmentation Sweep"),
    ]:
        _style_ax(ax)
        labels = df["option"].values
        x = np.arange(len(labels))
        w = 0.3
        for i, (metric, name, color) in enumerate([
            ("mean_map50", "mAP50", "#1D4ED8"),
            ("mean_b4_recall", "B4 Recall", "#10B981"),
        ]):
            bars = ax.bar(x + (i - 0.5) * w, df[metric], w, label=name, color=color, edgecolor="white")
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P2 / "figures" / "p2_batch_aug_sweep.png")


def f9_tuning_summary() -> None:
    df = pd.read_csv(P2 / "tuning_results.csv")
    row = df.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
    _style_ax(ax)
    labels = ["Phase 1 Baseline", "Best Tuned\n(lr0=0.0005)", "Final (reverted)"]
    values = [row["baseline_mean_map50"], row["tuned_mean_map50"], row["mean_map50"]]
    colors_list = ["#94A3B8", "#1D4ED8", "#F97316"]
    bars = ax.bar(labels, values, color=colors_list, edgecolor="white", width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#334155")
    ax.set_ylabel("Mean mAP50", fontsize=12)
    ax.set_title("Phase 2 Tuning — Progression mAP50", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.annotate("Reverted ke baseline\n(gain < 1%, tidak stabil)",
                xy=(2, values[2]), xytext=(1.8, values[2] - 0.015),
                fontsize=9, color="#DC2626", ha="center",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))
    ymin = min(values) - 0.02
    ymax = max(values) + 0.015
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    _save(fig, P2 / "figures" / "p2_tuning_summary.png")


def f16_imbalance_sweep() -> None:
    """Imbalance sweep — 3 identical bars proving loss is not the bottleneck."""
    df = pd.read_csv(P2 / "imbalance_sweep.csv")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)
    _style_ax(ax)
    labels = df["option"].values
    x = np.arange(len(labels))
    w = 0.22
    for i, (metric, name, color) in enumerate([
        ("mean_map50", "mAP50", "#1D4ED8"),
        ("mean_map50_95", "mAP50-95", "#059669"),
        ("mean_b4_recall", "B4 Recall", "#10B981"),
    ]):
        bars = ax.bar(x + (i - 1) * w, df[metric], w, label=name, color=color, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Loss Function Sweep — Semua Strategi Identik", fontsize=16, fontweight="bold", pad=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    # Annotate the identical result — centered banner below the chart title
    mid_y = (df["mean_map50"].iloc[0] + df["mean_b4_recall"].iloc[0]) / 2
    ax.text(1, mid_y, "Ketiga strategi menghasilkan\nmetrik yang persis sama",
            fontsize=10, color="#DC2626", ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#DC2626", lw=1.2, alpha=0.9))
    fig.tight_layout()
    _save(fig, P2 / "figures" / "p2_imbalance_sweep.png")


# ===================================================================
# Phase 3
# ===================================================================
def f10_per_class_metrics() -> None:
    df = _phase3_csv("per_class_metrics.csv")
    if df.empty:
        return
    required = {"branch", "checkpoint", "split", "candidate", "class_name", "precision", "recall", "map50", "map50_95"}
    if not required.issubset(df.columns):
        print("  [SKIP] per_class_metrics.csv still uses the old Phase 3 schema")
        return
    df = df[
        (df["branch"] == "one_stage")
        & (df["checkpoint"] == "last")
        & (df["split"] == "test")
    ].copy()
    if df.empty:
        print("  [SKIP] no one-stage per-class metrics for Phase 3")
        return

    candidates = _ordered_candidates(df["candidate"].dropna().unique().tolist())
    metrics_list = ["precision", "recall", "map50", "map50_95"]
    labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    colors_list = [
        METRIC_COLORS["precision"],
        METRIC_COLORS["recall"],
        METRIC_COLORS["mAP50"],
        METRIC_COLORS["mAP50-95"],
    ]

    fig, axes = plt.subplots(1, len(candidates), figsize=(6 * len(candidates), 6), dpi=DPI, sharey=True)
    if len(candidates) == 1:
        axes = [axes]

    for ax, candidate in zip(axes, candidates):
        _style_ax(ax)
        subset = df[df["candidate"] == candidate].copy()
        subset["class_name"] = pd.Categorical(subset["class_name"], categories=["B1", "B2", "B3", "B4"], ordered=True)
        subset = subset.sort_values("class_name")
        classes = subset["class_name"].tolist()
        x = np.arange(len(classes))
        width = 0.18
        for idx, (metric, label, color) in enumerate(zip(metrics_list, labels, colors_list)):
            vals = subset[metric].astype(float).tolist()
            bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=label, color=color, edgecolor="white")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=12, fontweight="bold")
        ax.set_title(f"{candidate} — last / test", fontsize=14, fontweight="bold", color=_candidate_color(candidate))
        ax.grid(axis="y", color="#E2E8F0", lw=0.8)
        ax.set_ylim(0, 0.98)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[-1].legend(fontsize=10, framealpha=0.9, ncol=2, loc="upper right")
    fig.suptitle("Metrik per Kelas — Kandidat Utama Phase 3", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_per_class_metrics.png")


def f11_threshold_sweep() -> None:
    df = _phase3_csv("threshold_sweep.csv")
    if df.empty:
        return
    required = {"branch", "checkpoint", "candidate", "conf", "precision", "recall", "map50", "map50_95"}
    if not required.issubset(df.columns):
        print("  [SKIP] threshold_sweep.csv still uses the old Phase 3 schema")
        return
    df = df[(df["branch"] == "one_stage") & (df["checkpoint"] == "last")].copy()
    if df.empty:
        print("  [SKIP] no threshold sweep rows for one-stage Phase 3")
        return
    df["conf"] = df["conf"].astype(float)
    candidates = _ordered_candidates(df["candidate"].dropna().unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=DPI, sharex=True)
    fig.suptitle("Threshold Sweep — One-Stage Phase 3 (`last`, val)", fontsize=17, fontweight="bold", y=1.02)
    metric_specs = [
        ("precision", "Precision", METRIC_COLORS["precision"]),
        ("recall", "Recall", METRIC_COLORS["recall"]),
        ("map50", "mAP50", METRIC_COLORS["mAP50"]),
        ("map50_95", "mAP50-95", METRIC_COLORS["mAP50-95"]),
    ]
    for ax, (metric, title, color) in zip(axes.flat, metric_specs):
        _style_ax(ax)
        for candidate in candidates:
            subset = df[df["candidate"] == candidate].sort_values("conf")
            ax.plot(
                subset["conf"],
                subset[metric].astype(float),
                marker="o",
                markersize=6,
                lw=2.2,
                color=_candidate_color(candidate),
                label=candidate,
                zorder=3,
            )
        ax.set_title(title, fontsize=13, fontweight="bold", color=color)
        ax.set_ylabel("Score", fontsize=11)
        ax.grid(color="#E2E8F0", lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Confidence Threshold", fontsize=11)
    axes[0, 1].legend(fontsize=10, framealpha=0.9)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_threshold_sweep_detail.png")


def f12_error_distribution() -> None:
    df = _phase3_csv("error_stratification.csv")
    if df.empty or "categories" not in df:
        return
    cat_counts: Counter = Counter()
    for cats in df["categories"].dropna():
        for cat in cats.split(";"):
            cat = cat.strip()
            if cat:
                cat_counts[cat] += 1
    cats_sorted = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)
    names = [c[0] for c in cats_sorted]
    counts = [c[1] for c in cats_sorted]
    colors_list = [ERROR_COLORS.get(n, "#94A3B8") for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=DPI)
    _style_ax(ax)
    bars = ax.barh(names, counts, color=colors_list, edgecolor="white", height=0.55)
    for bar in bars:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", va="center", fontsize=11, fontweight="bold", color="#334155")
    ax.set_xlabel("Jumlah Image", fontsize=12)
    ax.set_title("Distribusi Kategori Error — Top-20 Hardest Images", fontsize=16, fontweight="bold", pad=14)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E2E8F0", lw=0.8)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_error_distribution.png")


def f13_error_by_image() -> None:
    df = _phase3_csv("error_stratification.csv")
    if df.empty:
        return
    df["error_score"] = pd.to_numeric(df["error_score"], errors="coerce").fillna(0)
    df = df.sort_values("error_score", ascending=False).head(20)
    if df.empty:
        return
    dominant = df["categories"].apply(lambda x: x.split(";")[0].strip() if pd.notna(x) else "unknown")
    colors_list = [ERROR_COLORS.get(d, "#94A3B8") for d in dominant]
    short_names = df.apply(
        lambda row: f"{row.get('candidate', 'na')}:{Path(str(row.get('image_path', 'unknown'))).stem}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(14, 6), dpi=DPI)
    _style_ax(ax)
    bars = ax.bar(range(len(df)), df["error_score"], color=colors_list, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(short_names, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Error Score", fontsize=12)
    ax.set_title("Top-20 Image Tersulit by Error Score", fontsize=16, fontweight="bold", pad=14)
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"tp={row['tp']}\nfp={row['false_positive']}",
                ha="center", va="bottom", fontsize=7, color="#64748B")
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=n) for n, c in ERROR_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.9, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_error_by_image_score.png")


def f14_training_curves() -> None:
    """Training curves for the main one-stage Phase 3 candidates."""
    metrics_df = _phase3_one_stage_metrics(split="val", checkpoint="last")
    if metrics_df.empty:
        print("  [SKIP] no Phase 3 one-stage metrics available for training curves")
        return

    candidates = _ordered_candidates(metrics_df["candidate"].dropna().unique().tolist())
    run_names = {
        candidate: metrics_df[metrics_df["candidate"] == candidate]["run_name"].dropna().iloc[0]
        for candidate in candidates
    }
    curves: dict[str, pd.DataFrame] = {}
    for candidate, run_name in run_names.items():
        csv_path = RUNS / run_name / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        curves[candidate] = df
    if not curves:
        print("  [SKIP] missing training curves CSVs in runs/")
        return

    fig, axes = plt.subplots(2, len(curves), figsize=(7 * len(curves), 8), dpi=DPI, sharex="col")
    if len(curves) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle("Training Curves — Phase 3 One-Stage Candidates", fontsize=18, fontweight="bold", y=1.02)

    for col_idx, candidate in enumerate(candidates):
        if candidate not in curves:
            continue
        df = curves[candidate]
        epochs = df["epoch"]

        ax = axes[0, col_idx]
        _style_ax(ax)
        ax.plot(epochs, df["metrics/mAP50(B)"], lw=2.5, color=METRIC_COLORS["mAP50"], label="mAP50", zorder=3)
        ax.plot(epochs, df["metrics/mAP50-95(B)"], lw=2.5, color=METRIC_COLORS["mAP50-95"], label="mAP50-95", zorder=3)
        ax.set_title(f"{candidate} — Validation mAP", fontsize=14, fontweight="bold", color=_candidate_color(candidate))
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(color="#E2E8F0", lw=0.8)

        ax = axes[1, col_idx]
        _style_ax(ax)
        ax.plot(epochs, df["metrics/precision(B)"], lw=2.2, color=METRIC_COLORS["precision"], label="Precision", zorder=3)
        ax.plot(epochs, df["metrics/recall(B)"], lw=2.2, color=METRIC_COLORS["recall"], label="Recall", zorder=3)
        for loss_col, loss_label, loss_color in [
            ("train/box_loss", "Box Loss", "#94A3B8"),
            ("train/cls_loss", "Cls Loss", "#64748B"),
        ]:
            if loss_col in df:
                ax.plot(epochs, df[loss_col], lw=1.6, color=loss_color, label=loss_label, alpha=0.65)
        ax.set_title(f"{candidate} — Precision / Recall / Loss", fontsize=14, fontweight="bold", color=_candidate_color(candidate))
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Score / Loss", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9, ncol=2)
        ax.grid(color="#E2E8F0", lw=0.8)

    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_training_curves.png")


def f17_cross_phase_comparison() -> None:
    """Compare val vs test and candidate vs candidate for the main one-stage branch."""
    df = _phase3_csv("final_metrics.csv")
    if df.empty:
        return
    required = {"branch", "checkpoint", "candidate", "split", "map50", "map50_95", "precision", "recall"}
    if not required.issubset(df.columns):
        print("  [SKIP] final_metrics.csv still uses the old Phase 3 schema")
        return
    df = df[(df["branch"] == "one_stage") & (df["checkpoint"] == "last")].copy()
    if df.empty:
        print("  [SKIP] no one-stage Phase 3 rows for comparison chart")
        return

    candidates = _ordered_candidates(df["candidate"].dropna().unique().tolist())
    metric_specs = [
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("precision", "Precision"),
        ("recall", "Recall"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=DPI)
    fig.suptitle("Perbandingan Kandidat Utama Phase 3 (`last`)", fontsize=17, fontweight="bold", y=1.02)

    split_order = ["val", "test"]
    for ax, metrics_group, title in zip(
        axes,
        [metric_specs[:2], metric_specs[2:]],
        ["mAP Metrics", "Precision / Recall"],
    ):
        _style_ax(ax)
        labels = [f"{candidate}\n{split}" for candidate in candidates for split in split_order]
        x = np.arange(len(labels))
        width = 0.35 if len(metrics_group) == 2 else 0.28
        for idx, (metric, metric_label) in enumerate(metrics_group):
            vals = []
            for candidate in candidates:
                for split in split_order:
                    row = df[(df["candidate"] == candidate) & (df["split"] == split)]
                    vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
            bars = ax.bar(
                x + (idx - (len(metrics_group) - 1) / 2) * width,
                vals,
                width,
                label=metric_label,
                color=_metric_color(metric),
                edgecolor="white",
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=_metric_color(metric),
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", color="#E2E8F0", lw=0.8)
        ax.legend(fontsize=9, framealpha=0.9)
    axes[0].set_ylabel("Score", fontsize=12)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_cross_phase_comparison.png")


def f18_confusion_heatmaps() -> None:
    df = _phase3_csv("confusion_matrix.csv")
    if df.empty:
        return
    required = {"branch", "candidate", "checkpoint", "split", "true_class"}
    if not required.issubset(df.columns):
        print("  [SKIP] confusion_matrix.csv still uses the old Phase 3 schema")
        return
    class_order = _canonical_class_order(df.columns.tolist())
    if not class_order:
        print("  [SKIP] no class columns found in confusion_matrix.csv")
        return

    group_cols = ["branch", "candidate", "checkpoint", "split"]
    for keys, group in df.groupby(group_cols):
        branch, candidate, checkpoint, split = keys
        pivot = group.set_index("true_class").reindex(class_order)
        if pivot[class_order].isna().all().all():
            continue
        counts = pivot[class_order].fillna(0).astype(float).values
        fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=DPI)
        im = ax.imshow(counts, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(class_order)))
        ax.set_xticklabels(class_order, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(class_order)))
        ax.set_yticklabels(class_order, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("Ground Truth Class", fontsize=11)
        ax.set_title(f"{branch} / {candidate} / {checkpoint} / {split}", fontsize=12, fontweight="bold")
        for row_idx in range(len(class_order)):
            support = float(pivot["support"].iloc[row_idx]) if "support" in pivot else max(counts[row_idx].sum(), 1.0)
            for col_idx in range(len(class_order)):
                count = counts[row_idx, col_idx]
                pct = (count / support) if support else 0.0
                color = "white" if count >= counts.max() * 0.55 else "#1E293B"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{int(count)}\n{pct:.1%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02, label="Count")
        fig.tight_layout()
        out_name = f"cm_{branch}_{candidate}_{checkpoint}_{split}.png".replace("/", "_")
        _save(fig, P3 / "figures" / "confusion" / out_name)


def f19_checkpoint_comparison() -> None:
    df = _phase3_csv("final_metrics.csv")
    if df.empty:
        return
    required = {"branch", "candidate", "checkpoint", "split", "map50", "map50_95", "weighted_f1"}
    if not required.issubset(df.columns):
        print("  [SKIP] final_metrics.csv still uses the old Phase 3 schema")
        return
    df = df[df["branch"] == "one_stage"].copy()
    if df.empty:
        print("  [SKIP] no one-stage rows for checkpoint comparison")
        return
    if "best" not in set(df["checkpoint"].dropna().astype(str)):
        print("  [SKIP] checkpoint comparison requires both best and last rows")
        return

    candidates = _ordered_candidates(df["candidate"].dropna().unique().tolist())
    split_order = ["val", "test"]
    checkpoint_order = ["best", "last"]
    metric_specs = [
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
        ("weighted_f1", "Weighted F1"),
    ]

    fig, axes = plt.subplots(1, len(candidates), figsize=(6.4 * len(candidates), 6), dpi=DPI, sharey=True)
    if len(candidates) == 1:
        axes = [axes]
    fig.suptitle("Best vs Last pada Val dan Test — Kandidat One-Stage Phase 3", fontsize=17, fontweight="bold", y=1.02)

    for ax, candidate in zip(axes, candidates):
        _style_ax(ax)
        labels = [f"{checkpoint}\n{split}" for checkpoint in checkpoint_order for split in split_order]
        x = np.arange(len(labels))
        width = 0.22
        for idx, (metric, label) in enumerate(metric_specs):
            vals = []
            for checkpoint in checkpoint_order:
                for split in split_order:
                    row = _phase3_pick_row(df, branch="one_stage", candidate=candidate, checkpoint=checkpoint, split=split)
                    vals.append(float(row[metric]) if row is not None and pd.notna(row[metric]) else 0.0)
            bars = ax.bar(
                x + (idx - 1) * width,
                vals,
                width,
                color=_metric_color(metric),
                edgecolor="white",
                label=label,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color=_metric_color(metric),
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 0.75)
        ax.set_title(candidate, fontsize=14, fontweight="bold", color=_candidate_color(candidate))
        ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    axes[0].set_ylabel("Score", fontsize=12)
    axes[-1].legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_checkpoint_comparison.png")


def f20_pipeline_reference() -> None:
    df = _phase3_csv("final_metrics.csv")
    if df.empty:
        return
    required = {"branch", "candidate", "checkpoint", "split", "map50", "weighted_f1", "top1_acc", "precision", "recall"}
    if not required.issubset(df.columns):
        print("  [SKIP] final_metrics.csv still uses the old Phase 3 schema")
        return

    stage1_row = _phase3_pick_row(df, branch="two_stage_stage1", checkpoint="last", split="test")
    gtcrop_row = _phase3_pick_row(df, branch="two_stage_gtcrop", checkpoint="last", split="test")
    e2e_row = _phase3_pick_row(df, branch="two_stage_end_to_end", checkpoint="last", split="test")
    one_stage_rows = df[
        (df["branch"] == "one_stage")
        & (df["checkpoint"] == "last")
        & (df["split"] == "test")
    ].copy()
    if one_stage_rows.empty or stage1_row is None or gtcrop_row is None or e2e_row is None:
        print("  [SKIP] incomplete Phase 3 rows for pipeline reference")
        return
    one_stage_rows["map50"] = pd.to_numeric(one_stage_rows["map50"], errors="coerce")
    one_stage_rows["weighted_f1"] = pd.to_numeric(one_stage_rows["weighted_f1"], errors="coerce")
    one_stage_rows = one_stage_rows.sort_values("map50", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
    fig.suptitle("Ringkasan Branch Final Phase 3", fontsize=17, fontweight="bold", y=1.02)

    ax = axes[0]
    _style_ax(ax)
    labels = [
        f"One-stage\n{row['candidate']}\nmAP50" for _, row in one_stage_rows.iterrows()
    ] + [
        "Two-stage\nStage1\nmAP50",
        "Two-stage\nGT-crop\nTop-1",
        "Two-stage\nEnd-to-End\nWeighted F1",
    ]
    values = [float(row["map50"]) for _, row in one_stage_rows.iterrows()] + [
        float(stage1_row["map50"]),
        float(gtcrop_row["top1_acc"]),
        float(e2e_row["weighted_f1"]),
    ]
    colors_list = [
        _candidate_color(row["candidate"]) for _, row in one_stage_rows.iterrows()
    ] + ["#F97316", "#10B981", "#F97316"]
    bars = ax.bar(labels, values, color=colors_list, edgecolor="white", width=0.62)
    if len(bars) >= 1:
        bars[-2].set_hatch("//")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#334155",
        )
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_title("Metrik Referensi per Branch (`last`, test)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)

    ax = axes[1]
    _style_ax(ax)
    deploy_rows = one_stage_rows.copy()
    deploy_rows["label"] = deploy_rows["candidate"].apply(lambda value: f"One-stage\n{value}")
    deploy_rows = pd.concat([
        deploy_rows[["label", "precision", "recall", "weighted_f1"]],
        pd.DataFrame([{
            "label": "Two-stage\nEnd-to-End",
            "precision": float(e2e_row["precision"]),
            "recall": float(e2e_row["recall"]),
            "weighted_f1": float(e2e_row["weighted_f1"]),
        }]),
    ], ignore_index=True)
    x = np.arange(len(deploy_rows))
    width = 0.24
    for idx, (metric, label) in enumerate([
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("weighted_f1", "Weighted F1"),
    ]):
        bars = ax.bar(
            x + (idx - 1) * width,
            deploy_rows[metric].astype(float),
            width,
            color=_metric_color(metric),
            edgecolor="white",
            label=label,
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=_metric_color(metric),
            )
    ax.set_xticks(x)
    ax.set_xticklabels(deploy_rows["label"].tolist(), fontsize=10)
    ax.set_ylim(0, 0.72)
    ax.set_title("Metrik Operasional End-to-End (`last`, test)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#E2E8F0", lw=0.8)
    ax.legend(fontsize=9, framealpha=0.9)

    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(facecolor="#1D4ED8", label="Kandidat one-stage"),
        Patch(facecolor="#F97316", label="Komponen two-stage deployed"),
        Patch(facecolor="#10B981", hatch="//", label="GT-crop upper bound"),
    ], fontsize=9, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_pipeline_reference.png")


def f21_confusion_overview() -> None:
    df = _phase3_csv("confusion_matrix.csv")
    if df.empty:
        return
    required = {"branch", "candidate", "checkpoint", "split", "true_class", "support"}
    if not required.issubset(df.columns):
        print("  [SKIP] confusion_matrix.csv still uses the old Phase 3 schema")
        return

    gtcrop_candidate = df[df["branch"] == "two_stage_gtcrop"]["candidate"].dropna().astype(str).head(1)
    e2e_candidate = df[df["branch"] == "two_stage_end_to_end"]["candidate"].dropna().astype(str).head(1)
    if gtcrop_candidate.empty or e2e_candidate.empty:
        print("  [SKIP] missing two-stage confusion rows")
        return

    targets = [
        ("one_stage", "yolo11m", "last", "test", "One-stage yolo11m"),
        ("one_stage", "yolov8s", "last", "test", "One-stage yolov8s"),
        ("two_stage_gtcrop", gtcrop_candidate.iloc[0], "last", "test", "Two-stage GT-crop"),
        ("two_stage_end_to_end", e2e_candidate.iloc[0], "last", "test", "Two-stage end-to-end"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=DPI)
    fig.suptitle("Confusion Matrix 4 Kelas — Ringkasan Phase 3 (`last`, test)", fontsize=17, fontweight="bold", y=1.02)

    for ax, (branch, candidate, checkpoint, split, title) in zip(axes.flat, targets):
        _style_ax(ax)
        class_order, matrix = _phase3_confusion_counts(
            df,
            branch=branch,
            candidate=candidate,
            checkpoint=checkpoint,
            split=split,
        )
        if class_order is None or matrix is None:
            ax.axis("off")
            continue
        counts = matrix[class_order].fillna(0).astype(float).values
        support = matrix["support"].fillna(0).astype(float).values.reshape(-1, 1)
        norm = np.divide(counts, np.where(support == 0, 1.0, support))
        im = ax.imshow(norm, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(0.65, float(norm.max())))
        ax.set_xticks(range(len(class_order)))
        ax.set_xticklabels(class_order, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(class_order)))
        ax.set_yticklabels(class_order, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Ground Truth", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", color=_candidate_color(candidate))
        for row_idx in range(len(class_order)):
            for col_idx in range(len(class_order)):
                pct = norm[row_idx, col_idx]
                count = int(counts[row_idx, col_idx])
                color = "white" if pct >= 0.45 else "#1E293B"
                ax.text(
                    col_idx,
                    row_idx,
                    f"{count}\n{pct:.1%}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                    color=color,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save(fig, P3 / "figures" / "p3_confusion_overview.png")


# ===================================================================
# Main
# ===================================================================
PHASE_FUNCS = {
    0: [f1_class_distribution, f2_bbox_size, f3_resolution, f4_learning_curve],
    1: [f5_architecture_benchmark, f6_one_vs_two_stage, f15_per_class_heatmap],
    2: [f7_lr_sweep, f8_batch_aug_sweep, f9_tuning_summary, f16_imbalance_sweep],
    3: [f10_per_class_metrics, f11_threshold_sweep, f12_error_distribution,
        f13_error_by_image, f14_training_curves, f17_cross_phase_comparison,
        f18_confusion_heatmaps, f19_checkpoint_comparison, f20_pipeline_reference,
        f21_confusion_overview],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", nargs="*", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    for phase in args.phase:
        print(f"\n=== Phase {phase} ===")
        for fn in PHASE_FUNCS[phase]:
            fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
