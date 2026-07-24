"""
MutualFundDrift — All chart/visualisation functions in one file.
Contains 6 chart functions used by the backend API routers and Streamlit frontend.
All functions return a matplotlib Figure and never call plt.show() or plt.savefig().
"""

import logging
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Shared colour palette ─────────────────────────────────────────────────────
C_NORMAL = "#2ecc71"
C_WATCH  = "#f1c40f"
C_AMBER  = "#e67e22"
C_RED    = "#e74c3c"
C_LARGE  = "#3498db"
C_MID    = "#e67e22"
C_SMALL  = "#e74c3c"
C_CENTER = "#8e44ad"


def _severity_colour(drift: float) -> str:
    """Return a hex colour based on drift score severity."""
    if drift < 0.15: return C_NORMAL
    if drift < 0.25: return C_WATCH
    if drift < 0.35: return C_AMBER
    return C_RED


def _quarter_label(date_str: str) -> str:
    """Convert a YYYY-MM-DD string to 'Q1 2024' format."""
    try:
        p = str(date_str).split("-")
        return f"Q{(int(p[1])-1)//3+1} {p[0]}"
    except Exception:
        return str(date_str)


# ── 1. Style Box Journey ──────────────────────────────────────────────────────

def plot_style_box_journey(
    snapshots: List[dict],
    mandate: dict,
    scheme_name: str,
) -> plt.Figure:
    """
    Draw a 3x3 Morningstar style box and plot the fund's quarterly journey.

    Args:
        snapshots: List of snapshot dicts with size_score, style_score, drift_score, snapshot_date.
        mandate:   Dict with mandate_size_score and mandate_style_score.
        scheme_name: Fund name for the chart title.
    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 7))

    ms  = mandate.get("mandate_size_score",  0.5)
    mst = mandate.get("mandate_style_score", 0.5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.invert_yaxis()

    for v in [0.33, 0.67]:
        ax.axvline(v, color="#bdc3c7", linewidth=1.2, zorder=1)
        ax.axhline(v, color="#bdc3c7", linewidth=1.2, zorder=1)

    for si, sy in enumerate([0.17, 0.5, 0.83]):
        for sj, sx in enumerate([0.17, 0.5, 0.83]):
            ax.text(sx, sy, f"{['Large','Mid','Small'][si]}\n{['Value','Blend','Growth'][sj]}",
                    ha="center", va="center", fontsize=7, color="#95a5a6", alpha=0.7)

    def _cell(s):
        return (0.67,1.0) if s>0.67 else (0.33,0.67) if s>=0.33 else (0.0,0.33)

    mx0,mx1 = _cell(mst); my0,my1 = _cell(ms)
    ax.add_patch(mpatches.FancyBboxPatch((mx0,my0), mx0-mx0+mx1-mx0, my1-my0,
        boxstyle="round,pad=0.005", linewidth=0, facecolor=C_NORMAL, alpha=0.20, zorder=2))
    ax.plot(mst, ms, marker="*", markersize=14, color=C_NORMAL, zorder=5,
            label="Mandate Target", linewidth=0)

    if snapshots:
        xs = [s.get("style_score",0.5) for s in snapshots]
        ys = [s.get("size_score", 0.5) for s in snapshots]
        ds = [s.get("drift_score",0.0) for s in snapshots]
        n  = len(snapshots)
        colours = [plt.cm.Blues(0.35 + 0.65*(i/max(n-1,1))) for i in range(n)]

        ax.plot(xs, ys, color="#5d6d7e", linewidth=1.5, alpha=0.7, zorder=3, linestyle="--")
        for i,(x,y,c) in enumerate(zip(xs,ys,colours)):
            ax.scatter(x, y, color=c, s=180 if i==n-1 else 60, zorder=4,
                       edgecolors="white", linewidths=0.8)

        ax.annotate(f"Now\n(drift={ds[-1]:.2f})", xy=(xs[-1],ys[-1]),
                    xytext=(xs[-1]+0.08, ys[-1]-0.08), fontsize=8, color="#2c3e50",
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.0))
        ax.annotate(f"Q1\n({n} qtrs ago)", xy=(xs[0],ys[0]),
                    xytext=(xs[0]-0.1, ys[0]+0.1), fontsize=7, color="#7f8c8d",
                    arrowprops=dict(arrowstyle="->", color="#bdc3c7", lw=0.8))

    ax.set_xlabel("Style  (← Value  |  Growth →)", fontsize=10)
    ax.set_ylabel("Size  (↑ Large  |  Small ↓)",   fontsize=10)
    ax.set_xticks([0,0.33,0.67,1]); ax.set_xticklabels(["0.0\nValue","0.33","0.67","1.0\nGrowth"],fontsize=8)
    ax.set_yticks([0,0.33,0.67,1]); ax.set_yticklabels(["0.0\nLarge","0.33","0.67","1.0\nSmall"],fontsize=8)
    ax.set_title(f"{scheme_name}\nStyle Box Journey — Last {len(snapshots)} Quarter(s)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ── 2. Market Cap Composition ─────────────────────────────────────────────────

def plot_market_cap_composition(
    snapshots: List[dict],
    scheme_name: str,
) -> plt.Figure:
    """
    Stacked bar chart of large/mid/small cap allocation per quarter.

    Args:
        snapshots: List of snapshot dicts with large_cap_pct, mid_cap_pct, small_cap_pct, snapshot_date.
        scheme_name: Fund name for the chart title.
    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(10, 5))

    if not snapshots:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        ax.set_title(f"{scheme_name} — Market Cap Composition", fontsize=12, fontweight="bold")
        return fig

    labels = [_quarter_label(s.get("snapshot_date","")) for s in snapshots]
    large  = np.array([s.get("large_cap_pct",0) for s in snapshots])
    mid    = np.array([s.get("mid_cap_pct",  0) for s in snapshots])
    small  = np.array([s.get("small_cap_pct",0) for s in snapshots])
    x      = np.arange(len(snapshots))

    ax.bar(x, large, width=0.55, color=C_LARGE, label="Large Cap", alpha=0.88)
    ax.bar(x, mid,   width=0.55, bottom=large,       color=C_MID,   label="Mid Cap",   alpha=0.88)
    ax.bar(x, small, width=0.55, bottom=large+mid,   color=C_SMALL, label="Small Cap", alpha=0.88)

    for i,(lp,mp,sp) in enumerate(zip(large,mid,small)):
        for val, bot, lbl in [(lp,0,f"{lp:.0f}%"),(mp,lp,f"{mp:.0f}%"),(sp,lp+mp,f"{sp:.0f}%")]:
            if val > 8:
                ax.text(x[i], bot+val/2, lbl, ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("% of NAV", fontsize=10)
    ax.set_title(f"{scheme_name}\nMarket Cap Composition Over Time", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ── 3. Rolling Correlation Heatmap ────────────────────────────────────────────

def plot_rolling_correlation_heatmap(
    correlation_data: pd.DataFrame,
    scheme_name: str,
) -> plt.Figure:
    """
    Heatmap of rolling 12-month benchmark correlation (Quarter × Year).

    Args:
        correlation_data: DataFrame with columns snapshot_date and rolling_corr.
        scheme_name: Fund name for the chart title.
    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="white", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(10, 4))

    if correlation_data is None or correlation_data.empty:
        ax.text(0.5, 0.5, "No correlation data available", ha="center", va="center", fontsize=13)
        ax.set_title(f"{scheme_name} — Rolling Correlation", fontsize=12, fontweight="bold")
        return fig

    df = correlation_data.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["rolling_corr"]  = pd.to_numeric(df["rolling_corr"], errors="coerce")
    df = df.dropna(subset=["snapshot_date","rolling_corr"])

    if df.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", fontsize=13)
        return fig

    df["year"]    = df["snapshot_date"].dt.year
    df["quarter"] = df["snapshot_date"].dt.quarter.map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    pivot = df.pivot_table(index="quarter", columns="year", values="rolling_corr", aggfunc="mean")
    pivot = pivot.reindex([q for q in ["Q1","Q2","Q3","Q4"] if q in pivot.index])

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=0.0, vmax=1.0, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#ecf0f1", annot_kws={"size":9},
                cbar_kws={"label":"Rolling Correlation","shrink":0.8})
    ax.set_title(f"{scheme_name}\nRolling 12-Month Benchmark Correlation",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Year", fontsize=10); ax.set_ylabel("Quarter", fontsize=10)
    ax.tick_params(axis="x", labelrotation=0); ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    return fig


# ── 4. Sector Drift Heatmap ───────────────────────────────────────────────────

def plot_sector_drift_heatmap(
    sector_data: pd.DataFrame,
    scheme_name: str,
) -> plt.Figure:
    """
    Heatmap of sector weight (% NAV) across quarters, sorted by total weight.

    Args:
        sector_data: DataFrame with sectors as rows, quarters as columns, % weights as values.
        scheme_name: Fund name for the chart title.
    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="white", font_scale=0.95)

    if sector_data is None or sector_data.empty:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.text(0.5, 0.5, "No sector data available", ha="center", va="center", fontsize=13)
        ax.set_title(f"{scheme_name} — Sector Drift Heatmap", fontsize=12, fontweight="bold")
        return fig

    df = sector_data.copy()
    df["_total"] = df.sum(axis=1)
    df = df.sort_values("_total", ascending=False).drop(columns="_total")

    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(max(8, n_cols*1.1+2), max(4, n_rows*0.55+1.5)))

    sns.heatmap(df, ax=ax, cmap="YlOrRd", vmin=0,
                vmax=max(35, df.values.max() if df.values.size>0 else 35),
                annot=True, fmt=".1f", linewidths=0.4, linecolor="#ecf0f1",
                annot_kws={"size":8}, cbar_kws={"label":"% of NAV","shrink":0.7})

    if n_cols > 2:
        ax.axvline(x=n_cols-2, color="#2c3e50", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(n_cols-2+0.05, -0.4, "Recent", fontsize=8, color="#2c3e50",
                transform=ax.get_xaxis_transform(), ha="left")

    ax.set_title(f"{scheme_name}\nSector Weight Drift Heatmap", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Quarter", fontsize=10); ax.set_ylabel("GICS Sector", fontsize=10)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
    fig.tight_layout()
    return fig


# ── 5. Drift Score Timeline ───────────────────────────────────────────────────

def plot_drift_timeline(
    snapshots: List[dict],
    threshold: float,
    predictions: List[dict],
    scheme_name: str,
) -> plt.Figure:
    """
    Line chart of actual and predicted drift scores with severity bands.

    Args:
        snapshots:   List of snapshot dicts with snapshot_date and drift_score.
        threshold:   Alert threshold reference line value.
        predictions: List of prediction dicts with prediction_date and predicted_drift_score.
        scheme_name: Fund name for the chart title.
    Returns:
        matplotlib Figure.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(11, 5))

    if not snapshots:
        ax.text(0.5, 0.5, "No drift data available", ha="center", va="center", fontsize=13)
        ax.set_title(f"{scheme_name} — Drift Score Timeline", fontsize=12, fontweight="bold")
        return fig

    actual = pd.DataFrame([
        {"date": pd.to_datetime(s["snapshot_date"]), "drift": s["drift_score"]}
        for s in snapshots if "snapshot_date" in s and "drift_score" in s
    ]).sort_values("date")

    for y0,y1,col,lbl in [(0,0.15,C_NORMAL,"Normal (<0.15)"),
                           (0.15,0.25,C_WATCH, "Watch (0.15–0.25)"),
                           (0.25,0.35,C_AMBER, "Amber (0.25–0.35)"),
                           (0.35,1.50,C_RED,   "Red Alert (>0.35)")]:
        ax.axhspan(y0, y1, alpha=0.12, color=col, zorder=0, label=lbl)

    if not actual.empty:
        ax.plot(actual["date"], actual["drift"], color="#2980b9", linewidth=2.2,
                marker="o", markersize=5, label="Actual Drift Score", zorder=4)
        last = actual.iloc[-1]
        ax.annotate(f"  {last['drift']:.3f}", xy=(last["date"],last["drift"]),
                    fontsize=9, color="#2980b9", fontweight="bold")

    if predictions:
        pred = pd.DataFrame([
            {"date": pd.to_datetime(p["prediction_date"]), "val": p["predicted_drift_score"]}
            for p in predictions if "prediction_date" in p
        ]).sort_values("date")
        if not pred.empty:
            ax.plot(pred["date"], pred["val"], color=C_AMBER, linewidth=1.8,
                    linestyle="--", marker="s", markersize=4, label="Predicted Drift", zorder=4)
            ax.annotate(f"  {pred.iloc[-1]['val']:.3f}", xy=(pred.iloc[-1]["date"], pred.iloc[-1]["val"]),
                        fontsize=8.5, color=C_AMBER)

    ax.axhline(threshold, color="#c0392b", linestyle=":", linewidth=1.5, alpha=0.8,
               label=f"Alert Threshold ({threshold})")
    ax.set_ylim(0, 1.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Date", fontsize=10); ax.set_ylabel("Drift Score", fontsize=10)
    ax.set_title(f"{scheme_name}\nDrift Score Timeline", fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)
    ax.yaxis.grid(True, alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout()
    return fig

