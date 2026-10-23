"""Static figures for the scouting report (matplotlib)."""
from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.similarity import FEATURE_LABEL

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#a3a29d"
GRID = "#e6e5e1"
BLUE = "#2a78d6"      # series 1
ORANGE = "#eb6834"    # series 2

# radar layout: features grouped into four themes, clockwise
RADAR_GROUPS = {
    "Shooting": ["np_xg", "shots", "touches_in_box"],
    "Creating": ["xa", "key_passes", "passes_into_box", "crosses"],
    "Progressing": ["progressive_passes", "final_third_passes", "progressive_carries", "dribbles_completed", "pass_completion"],
    "Defending": ["tackles_won_padj", "interceptions_padj", "pressures_padj", "recoveries", "clearances"],
}
RADAR_FEATURES = [f for fs in RADAR_GROUPS.values() for f in fs]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _polar_base(ax, n):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([])
    ax.yaxis.grid(color=GRID, lw=0.8)
    ax.xaxis.grid(color=GRID, lw=0.8)
    ax.spines["polar"].set_color(GRID)
    ax.set_xticks(angles)
    ax.set_xticklabels([FEATURE_LABEL[f] for f in RADAR_FEATURES], fontsize=8.5, color=INK_2)
    for lbl, a in zip(ax.get_xticklabels(), angles):
        deg = np.degrees(a)
        lbl.set_horizontalalignment("left" if 5 < deg < 175 else "right" if 185 < deg < 355 else "center")
    ax.tick_params(axis="x", pad=14)
    return angles


def pizza(model, name: str, path=None):
    """Percentile 'pizza' for one player, vs players in the same position group."""
    pid = model.find(name)
    me = model.players.loc[pid]
    pct = model.percentiles.loc[pid, RADAR_FEATURES].values
    fig, ax = plt.subplots(figsize=(7.2, 7.6), subplot_kw={"projection": "polar"})
    angles = _polar_base(ax, len(RADAR_FEATURES))
    width = 2 * np.pi / len(RADAR_FEATURES) * 0.86
    ax.bar(angles, pct, width=width, color=BLUE, alpha=0.9, edgecolor=SURFACE, linewidth=2)
    for a, v in zip(angles, pct):
        ax.text(a, v + 8, f"{v:.0f}", ha="center", va="center", fontsize=8, color=INK, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    fig.suptitle(me.player, fontsize=15, fontweight="bold", x=0.06, ha="left", y=0.98)
    fig.text(0.06, 0.925, f"{me.team} · {me.role} · {me.minutes:,.0f} min\n"
             f"Percentile vs Premier League {me.group} players, 2015/16 (≥900 min)",
             fontsize=9.5, color=INK_2, va="top")
    fig.text(0.06, 0.02, "Data: StatsBomb open data. PAdj = possession-adjusted.", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.80, bottom=0.08)
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def compare(model, a: str, b: str, path=None):
    """Two-player overlay radar (percentiles within each player's own group)."""
    fig, ax = plt.subplots(figsize=(7.2, 7.8), subplot_kw={"projection": "polar"})
    angles = _polar_base(ax, len(RADAR_FEATURES))
    closed = np.append(angles, angles[0])
    sim = model.similar(a, n=len(model.players), ).set_index("player").similarity
    for name, color in ((a, BLUE), (b, ORANGE)):
        pid = model.find(name)
        v = model.percentiles.loc[pid, RADAR_FEATURES].values
        v = np.append(v, v[0])
        ax.plot(closed, v, color=color, lw=2)
        ax.fill(closed, v, color=color, alpha=0.12)
    pa, pb = model.players.loc[model.find(a)], model.players.loc[model.find(b)]
    fig.suptitle(f"{pa.player} vs {pb.player}", fontsize=15, fontweight="bold", x=0.06, ha="left", y=0.98)
    score = sim.get(pb.player, np.nan)
    fig.text(0.06, 0.925, f"Style similarity {score:.1f}/100 · percentiles vs own position group",
             fontsize=9.5, color=INK_2, va="top")
    handles = [plt.Line2D([], [], color=BLUE, lw=2.5), plt.Line2D([], [], color=ORANGE, lw=2.5)]
    fig.legend(handles, [f"{pa.player} ({pa.team})", f"{pb.player} ({pb.team})"],
               loc="upper left", bbox_to_anchor=(0.05, 0.89), frameon=False, ncol=2, fontsize=9.5)
    fig.text(0.06, 0.02, "Data: StatsBomb open data, Premier League 2015/16.", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.78, bottom=0.08)
    if path:
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def role_map(model, path=None):
    """Small multiples: each role highlighted on the first two principal components."""
    P = model.players
    roles = model.role_summary().role.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True, sharey=True)
    ev = model.pca.explained_variance_ratio_
    for ax, role in zip(axes.flat, roles):
        g = P[P.role == role]
        ax.scatter(P.pc1, P.pc2, s=12, color=GRID, edgecolor="none")
        ax.scatter(g.pc1, g.pc2, s=26, color=BLUE, edgecolor=SURFACE, linewidth=1)
        for _, r in g.nlargest(3, "minutes").iterrows():
            ax.annotate(r.player, (r.pc1, r.pc2), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=INK)
        ax.set_title(f"{role}  ({len(g)})", fontsize=10.5, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)", fontsize=8.5)
    fig.suptitle("Eight playing-style roles found by K-Means", fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.935, "Each panel highlights one role on the first two principal components; "
             "grey dots are all other outfield players (≥900 min).", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def k_diagnostics(diag: pd.DataFrame, chosen: int, path=None):
    """Silhouette and inertia as two separate panels (never a dual axis)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, col, title in ((a1, "silhouette", "Silhouette score (higher = tighter)"),
                           (a2, "inertia", "Within-cluster inertia (elbow)")):
        ax.plot(diag.k, diag[col], color=BLUE, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
        row = diag[diag.k == chosen].iloc[0]
        ax.scatter([chosen], [row[col]], s=90, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"chosen k={chosen}", (chosen, row[col]), xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("k (number of roles)")
        ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def similar_bars(model, name: str, n: int = 10, path=None, **kw):
    s = model.similar(name, n=n, **kw).iloc[::-1]
    me = model.players.loc[model.find(name)]
    fig, ax = plt.subplots(figsize=(8, 0.42 * n + 1.4))
    y = np.arange(len(s))
    ax.set_axisbelow(True)
    ax.hlines(y, s.similarity.min() - 3, s.similarity, color=GRID, lw=1)
    ax.scatter(s.similarity, y, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for yy, v in zip(y, s.similarity):
        ax.text(v + 0.25, yy, f"{v:.1f}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(y, s.player + "  ·  " + s.team)
    ax.set_xlim(s.similarity.min() - 3, 100)
    ax.set_title(f"Most similar players to {me.player}", loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("Style similarity (0–100, cosine in PCA space)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9, labelcolor=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
