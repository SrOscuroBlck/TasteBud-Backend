"""
E4 — Fidelidad de Personalización
====================================
Verifies that the magnitude of recommendation differences between users
correlates with the magnitude of their profile differences.

Methods tested: BAYESIANO (always), SISTEMA (with --llm), POPULARIDAD (baseline).
15 unique user pairs (C(6,2)), Spearman rank correlation.

Usage:
    python scripts/evaluation/exp4_personalization_fidelity.py
    python scripts/evaluation/exp4_personalization_fidelity.py --llm
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from helpers import (
    FIGURES_DIR, RESULTS_DIR,
    apply_safety_filter, cosine_similarity, jaccard_distance, profile_distance,
    ensure_results_dir, get_db_session, get_evaluation_restaurant,
    save_csv, set_seed,
)
from user_profiles import SIMULATED_USERS, build_user

from services.core.reranking_service import RecommendationContext, RerankingService
from services.learning.bayesian_profile_service import BayesianProfileService


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation conditions
# ─────────────────────────────────────────────────────────────────────────────

def recs_bayesiano(safe_items, user, db_session, top_n=10):
    svc = BayesianProfileService()
    profile = svc.create_profile_from_user(db_session, user)
    context = RecommendationContext(time_of_day="afternoon")
    reranker = RerankingService(population_stats=None)
    ranked = reranker.rerank(safe_items, user, context, top_n=top_n, bayesian_profile=profile)
    return [str(r.item.id) for r in ranked]


def recs_popularidad(safe_items, top_n=10):
    ranked = sorted(safe_items, key=lambda it: (it.price or 0.0), reverse=True)
    return [str(it.id) for it in ranked[:top_n]]


def recs_sistema(safe_items, user, restaurant_id, top_n=10):
    from services.ml.llm_reranking_service import MAX_CANDIDATES_FOR_LLM, rerank_single_items
    from models.session import RecommendationSession

    candidates = sorted(
        safe_items,
        key=lambda it: cosine_similarity(user.taste_vector, it.features or {}),
        reverse=True,
    )[:MAX_CANDIDATES_FOR_LLM]

    items_map = {str(it.id): it for it in candidates}
    rec_session = RecommendationSession(
        id=uuid4(),
        user_id=user.id,
        restaurant_id=restaurant_id,
        meal_intent="main_only",
        time_of_day="afternoon",
        detected_hour=13,
        day_of_week=2,
    )

    results = rerank_single_items(
        candidates=candidates,
        user=user,
        rec_session=rec_session,
        feedback_list=[],
        items_map=items_map,
        top_n=top_n,
    )
    return [str(entry.get("item_id", "")) for entry in results if entry.get("item_id")][:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _scatter_subplot(ax, x_vals, y_vals, pair_labels, title, color):
    if not x_vals or all(v == 0 for v in y_vals):
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    rho, p_val = scipy_stats.spearmanr(x_vals, y_vals)

    ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=60, zorder=3)

    # Regression line
    slope, intercept, *_ = scipy_stats.linregress(x_vals, y_vals)
    x_line = np.linspace(min(x_vals), max(x_vals), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2,
            label=f"ρ = {rho:.2f}, p = {p_val:.3f}")

    for i, (lbl, xv, yv) in enumerate(zip(pair_labels, x_vals, y_vals)):
        ax.annotate(lbl, (xv, yv), fontsize=6, xytext=(4, 4),
                    textcoords="offset points", alpha=0.75)

    ax.set_xlabel("Distancia entre Perfiles de Sabor\n(1 − similitud coseno)")
    ax.set_ylabel("Distancia entre Recomendaciones\n(1 − Jaccard)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)


def plot_results(pairs_data, methods_present, output_path):
    n_subplots = len(methods_present)
    method_colors = {"BAYESIANO": "steelblue", "SISTEMA": "darkorange", "POPULARIDAD": "gray"}
    method_titles = {
        "BAYESIANO": "BAYESIANO — Fidelidad de Personalización",
        "SISTEMA": "SISTEMA (LLM) — Fidelidad de Personalización",
        "POPULARIDAD": "POPULARIDAD — Baseline (sin personalización)",
    }

    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    if n_subplots == 1:
        axes = [axes]

    x_vals = [d["profile_distance"] for d in pairs_data]
    pair_labels = [f"{d['user_i'][:3]}–{d['user_j'][:3]}" for d in pairs_data]

    for ax, method in zip(axes, methods_present):
        y_vals = [d.get(f"rec_distance_{method.lower()}", 0.0) for d in pairs_data]
        _scatter_subplot(ax, x_vals, y_vals, pair_labels,
                         method_titles[method], method_colors[method])

    plt.suptitle("¿El sistema personaliza proporcionalmente a las diferencias entre perfiles?",
                 fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E4 — Fidelidad de Personalización")
    parser.add_argument("--llm", action="store_true", help="Include SISTEMA condition (OpenAI API)")
    args = parser.parse_args()

    set_seed(42)
    ensure_results_dir()

    print("\n=== E4 — Fidelidad de Personalización ===\n")

    users = [build_user(p) for p in SIMULATED_USERS]
    user_short = [p["label"].split("—")[0].strip() for p in SIMULATED_USERS]

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items\n")

        # ── Get recommendations for each user under each condition ────────────
        bayesiano_recs = {}
        popularidad_recs = {}
        sistema_recs = {} if args.llm else None

        for user, profile in zip(users, SIMULATED_USERS):
            label = profile["label"]
            safe = apply_safety_filter(all_items, user)
            print(f"  {label}: {len(safe)} safe items")

            bayesiano_recs[label] = recs_bayesiano(safe, user, session)
            popularidad_recs[label] = recs_popularidad(safe)

            if args.llm:
                try:
                    sistema_recs[label] = recs_sistema(safe, user, restaurant_id)
                    print(f"    SISTEMA: {len(sistema_recs[label])} items")
                except Exception as e:
                    print(f"    SISTEMA ERROR: {e}")
                    sistema_recs[label] = []

        print()

        # ── Compute pairwise distances ────────────────────────────────────────
        rows = []
        pairs_data = []
        labels = [p["label"] for p in SIMULATED_USERS]

        for i, j in itertools.combinations(range(len(users)), 2):
            u_i, u_j = users[i], users[j]
            l_i, l_j = labels[i], labels[j]

            p_dist = profile_distance(u_i, u_j)
            r_dist_b = jaccard_distance(bayesiano_recs[l_i], bayesiano_recs[l_j])
            r_dist_p = jaccard_distance(popularidad_recs[l_i], popularidad_recs[l_j])
            r_dist_s = (
                jaccard_distance(sistema_recs.get(l_i, []), sistema_recs.get(l_j, []))
                if args.llm else None
            )

            row = {
                "user_i": user_short[i],
                "user_j": user_short[j],
                "profile_distance": round(p_dist, 6),
                "rec_distance_bayesiano": round(r_dist_b, 6),
                "rec_distance_popularidad": round(r_dist_p, 6),
            }
            pair_entry = dict(row)
            pair_entry["profile_distance"] = p_dist

            if args.llm and r_dist_s is not None:
                row["rec_distance_sistema"] = round(r_dist_s, 6)
                pair_entry["rec_distance_sistema"] = r_dist_s

            rows.append(row)
            pairs_data.append(pair_entry)
            print(f"  {user_short[i]:12s} × {user_short[j]:12s}  "
                  f"prof_dist={p_dist:.4f}  "
                  f"bay={r_dist_b:.4f}  pop={r_dist_p:.4f}"
                  + (f"  llm={r_dist_s:.4f}" if r_dist_s is not None else ""))

    save_csv(rows, RESULTS_DIR / "exp4_personalization.csv")

    methods_present = ["BAYESIANO", "POPULARIDAD"]
    if args.llm:
        methods_present = ["BAYESIANO", "SISTEMA", "POPULARIDAD"]

    plot_results(pairs_data, methods_present, FIGURES_DIR / "exp4_personalization_fidelity.png")

    # Print Spearman correlations summary
    print("\n  Spearman correlations:")
    x_vals = [d["profile_distance"] for d in pairs_data]
    for method in methods_present:
        y_key = f"rec_distance_{method.lower()}"
        y_vals = [d.get(y_key, 0.0) for d in pairs_data]
        if len(set(y_vals)) > 1:
            rho, p = scipy_stats.spearmanr(x_vals, y_vals)
            print(f"    {method:12s}: rho={rho:.3f}  p={p:.3f}")
        else:
            print(f"    {method:12s}: constant output (rho undefined)")

    print("\nE4 complete.")


if __name__ == "__main__":
    main()
