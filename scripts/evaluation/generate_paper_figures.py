"""
Generate English-language versions of the thesis figures for the academic paper.

Uses the EXACT same plotting code as the original experiment scripts
(exp1_ablation.py, exp2_coldstart.py, exp3_convergence.py, exp4_personalization_fidelity.py)
but with Spanish strings replaced by English equivalents.

Reads only from the existing results files — does NOT re-run any experiments.

Output: ../../../../paper/figures/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR     = Path(__file__).parent.parent.parent.parent / "paper" / "figures"
THESIS_FIGS = Path(__file__).parent.parent.parent.parent / "Tesis" / "USB_template" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Ablation (E1)  — same as exp1_ablation.plot_results() in English
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation():
    df = pd.read_csv(RESULTS_DIR / "exp1_ablation.csv")

    # Translate condition names
    cond_map = {
        "BAYESIANO":   "BAYESIAN",
        "COSENO":      "COSINE",
        "POPULARIDAD": "POPULARITY",
        "ALEATORIO":   "RANDOM",
        "SISTEMA":     "SYSTEM(LLM)",
    }
    df["condition"] = df["condition"].map(lambda c: cond_map.get(c, c))

    # Preserve user order and short names (same split logic as original)
    user_labels = df["user_label"].unique().tolist()
    # Keep the order matching SIMULATED_USERS
    desired_order = [
        "Carlos — Picante/Umami",
        "Valentina — Dulce/Suave",
        "Andres — Umami/Graso",
        "Maria — Vegana/Acida",
        "Santiago — Equilibrado/Explorador",
        "Isabella — Sin Gluten",
    ]
    user_labels = [u for u in desired_order if u in user_labels]
    conditions  = [c for c in ["BAYESIAN", "COSINE", "POPULARITY", "RANDOM", "SYSTEM(LLM)"]
                   if c in df["condition"].values]

    rows = df.to_dict("records")
    # Rename key to match original function expectation
    for r in rows:
        r["user_label"] = r.get("user_label", "")

    tas_data = {c: [] for c in conditions}
    ild_data = {c: [] for c in conditions}
    for label in user_labels:
        for cond in conditions:
            match = [r for r in rows if r["user_label"] == label and r["condition"] == cond]
            tas_data[cond].append(float(match[0]["tas_at_10"]) if match else 0.0)
            ild_data[cond].append(float(match[0]["ild_at_10"]) if match else 0.0)

    # Same colours as original script
    colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c", "#999999"]
    x     = np.arange(len(user_labels))
    width = 0.15

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    for ax_idx, (metric_data, ylabel, title) in enumerate([
        (tas_data, "TAS@10", "Taste Alignment Score by Method"),
        (ild_data, "ILD@10", "Intra-List Diversity by Method"),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        for i, (cond, color) in enumerate(zip(conditions, colors)):
            ax.bar(x + i * width, metric_data[cond], width, label=cond,
                   color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Simulated User Profile")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        short_labels = [lb.split("—")[0].strip() for lb in user_labels]
        ax.set_xticklabels(short_labels, rotation=15, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Recommendation Pipeline Ablation — TasteBud", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig_ablation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Cold-Start (E2)  — same as exp2_coldstart.plot_results() in English
# ─────────────────────────────────────────────────────────────────────────────

def fig_coldstart():
    df = pd.read_csv(RESULTS_DIR / "exp2_coldstart.csv")

    desired_order = [
        "Carlos — Picante/Umami",
        "Valentina — Dulce/Suave",
        "Andres — Umami/Graso",
        "Maria — Vegana/Acida",
        "Santiago — Equilibrado/Explorador",
        "Isabella — Sin Gluten",
    ]
    user_labels = [u for u in desired_order if u in df["user_label"].values]

    methods       = ["onboarding", "poblacional", "plano"]
    method_labels = ["Bayesian Onboarding", "Population Prior", "Flat Prior Beta(1,1)"]
    colors        = ["#1a9641", "#fdae61", "#d7191c"]   # same as original

    rows = df.to_dict("records")
    tas_data = {m: [] for m in methods}
    for label in user_labels:
        for m in methods:
            match = [r for r in rows if r["user_label"] == label and r["method"] == m]
            tas_data[m].append(float(match[0]["tas_at_10"]) if match else 0.0)

    x     = np.arange(len(user_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, color, ml) in enumerate(zip(methods, colors, method_labels)):
        ax.bar(x + i * width, tas_data[m], width, label=ml, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Simulated User Profile")
    ax.set_ylabel("TAS@10 at Session 0")
    ax.set_title("Cold-Start Recommendation Quality\nby Profile Initialization Method")
    ax.set_xticks(x + width)
    short_labels = [lb.split("—")[0].strip() for lb in user_labels]
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = OUT_DIR / "fig_coldstart.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Convergence (E3)  — same as exp3_convergence.plot_convergence() in English
# ─────────────────────────────────────────────────────────────────────────────

def fig_convergence():
    with open(RESULTS_DIR / "exp3_convergence.json") as f:
        all_data = json.load(f)

    profile_styles = [
        ("Carlos — Picante/Umami",  "spicy", "#e74c3c", 0.90),
        ("Valentina — Dulce/Suave", "sweet", "#e91e8c", 0.90),
        ("Andres — Umami/Graso",    "umami", "#2196f3", 0.90),
    ]

    n_sessions = max(len(v) for v in all_data.values())
    sessions   = list(range(1, n_sessions + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, axis, color, target in profile_styles:
        if label not in all_data:
            continue
        data    = all_data[label]
        e_theta = [d["e_theta"] for d in data]
        tas_vals = [d["tas_5"]  for d in data]

        short = label.split("—")[0].strip()
        axes[0].plot(sessions, e_theta, marker="o", color=color,
                     label=f"{short} ({axis})", linewidth=2)
        axes[0].axhline(y=target, color=color, linestyle="--", alpha=0.4, linewidth=1)

        axes[1].plot(sessions, tas_vals, marker="s", color=color,
                     label=short, linewidth=2)

    axes[0].set_xlabel("Feedback Session")
    axes[0].set_ylabel(r"E[$\theta$] — Estimated Preference (Dominant Axis)")
    axes[0].set_title("Bayesian Profile Convergence")
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_xticks(sessions)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Feedback Session")
    axes[1].set_ylabel("TAS@5")
    axes[1].set_title("Recommendation Quality per Session")
    axes[1].set_ylim(0.3, 1.0)
    axes[1].set_xticks(sessions)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle("System Learning Across Feedback Sessions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "fig_convergence.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Personalization Fidelity (E4)  — same as exp4.plot_results() in English
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_subplot_en(ax, x_vals, y_vals, pair_labels, title, color):
    if not x_vals or all(v == 0 for v in y_vals):
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    rho, p_val = scipy_stats.spearmanr(x_vals, y_vals)

    ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=60, zorder=3)

    slope, intercept, *_ = scipy_stats.linregress(x_vals, y_vals)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label=f"ρ = {rho:.2f}, p = {p_val:.3f}")

    for lbl, xv, yv in zip(pair_labels, x_vals, y_vals):
        ax.annotate(lbl, (xv, yv), fontsize=6, xytext=(4, 4),
                    textcoords="offset points", alpha=0.75)

    ax.set_xlabel("Taste Profile Distance\n(1 − cosine similarity)")
    ax.set_ylabel("Recommendation Distance\n(1 − Jaccard)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def fig_personalization():
    df = pd.read_csv(RESULTS_DIR / "exp4_personalization.csv")
    # Columns: user_i, user_j, profile_distance, rec_distance_bayesiano, rec_distance_popularidad

    pairs_data = df.to_dict("records")
    x_vals     = [d["profile_distance"] for d in pairs_data]
    pair_labels = [f"{str(d['user_i'])[:3]}–{str(d['user_j'])[:3]}" for d in pairs_data]

    methods_present = ["bayesiano", "popularidad"]
    method_colors = {"bayesiano": "steelblue", "popularidad": "gray"}
    method_titles = {
        "bayesiano":   "BAYESIAN — Personalization Fidelity",
        "popularidad": "POPULARITY — Baseline (no personalization)",
    }

    n_subplots = len(methods_present)
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    for ax, method in zip(axes, methods_present):
        col = f"rec_distance_{method}"
        y_vals = [d.get(col, 0.0) for d in pairs_data]
        _scatter_subplot_en(ax, x_vals, y_vals, pair_labels,
                            method_titles[method], method_colors[method])

    plt.suptitle("Does the system personalize proportionally to profile differences?",
                 fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "fig_personalization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Copy pipeline architecture diagram (no Spanish text)
# ─────────────────────────────────────────────────────────────────────────────

def copy_pipeline():
    src = THESIS_FIGS / "full-pipeline-flow.png"
    dst = OUT_DIR / "pipeline.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Copied {src.name} → {dst.name}")
    else:
        print(f"  WARNING: {src} not found — add manually.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures (English)...\n")
    fig_ablation()
    fig_coldstart()
    fig_convergence()
    fig_personalization()
    copy_pipeline()
    print(f"\nDone. Files in: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}")
