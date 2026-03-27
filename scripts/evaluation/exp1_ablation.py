"""
E1 — Ablación de Componentes del Pipeline
==========================================
Compares 5 recommendation conditions across 6 synthetic user profiles.
Metrics: TAS@10 (Taste Alignment Score) and ILD@10 (Intra-List Diversity).

Usage:
    python scripts/evaluation/exp1_ablation.py          # All conditions except SISTEMA
    python scripts/evaluation/exp1_ablation.py --llm    # Include SISTEMA (OpenAI API calls)
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from helpers import (
    FIGURES_DIR, RESULTS_DIR,
    apply_safety_filter, compute_ild, compute_tas, cosine_similarity,
    ensure_results_dir, get_db_session, get_evaluation_restaurant,
    save_csv, set_seed,
)
from user_profiles import SIMULATED_USERS, build_user

from models.session import RecommendationSession
from services.core.reranking_service import RecommendationContext, RerankingService
from services.learning.bayesian_profile_service import BayesianProfileService


# ─────────────────────────────────────────────────────────────────────────────
# Condition implementations
# ─────────────────────────────────────────────────────────────────────────────

def condition_coseno(safe_items, user, top_n=10):
    ranked = sorted(safe_items, key=lambda it: cosine_similarity(user.taste_vector, it.features or {}), reverse=True)
    return ranked[:top_n]


def condition_popularidad(safe_items, top_n=10):
    ranked = sorted(safe_items, key=lambda it: (it.price or 0.0), reverse=True)
    return ranked[:top_n]


def condition_aleatorio(safe_items, seed=42, top_n=10):
    pool = list(safe_items)
    random.seed(seed)
    random.shuffle(pool)
    return pool[:top_n]


def condition_bayesiano(safe_items, user, db_session, top_n=10):
    svc = BayesianProfileService()
    profile = svc.create_profile_from_user(db_session, user)
    context = RecommendationContext(time_of_day="afternoon")
    reranker = RerankingService(population_stats=None)
    ranked = reranker.rerank(safe_items, user, context, top_n=top_n, bayesian_profile=profile)
    return [r.item for r in ranked]


def condition_sistema(safe_items, user, restaurant_id, top_n=10):
    """LLM reranking via rerank_single_items(). Requires OPENAI_API_KEY."""
    from services.ml.llm_reranking_service import (
        MAX_CANDIDATES_FOR_LLM, rerank_single_items,
    )
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

    ordered = []
    for entry in results:
        item = items_map.get(str(entry.get("item_id", "")))
        if item:
            ordered.append(item)
    return ordered[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(rows, conditions, user_labels, output_path):
    tas_data = {c: [] for c in conditions}
    ild_data = {c: [] for c in conditions}

    for label in user_labels:
        for cond in conditions:
            match = [r for r in rows if r["user_label"] == label and r["condition"] == cond]
            tas_data[cond].append(float(match[0]["tas_at_10"]) if match else 0.0)
            ild_data[cond].append(float(match[0]["ild_at_10"]) if match else 0.0)

    colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c", "#999999"]
    x = np.arange(len(user_labels))
    width = 0.15

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    for ax_idx, (metric_data, ylabel, title) in enumerate([
        (tas_data, "TAS@10", "Taste Alignment Score por Método"),
        (ild_data, "ILD@10", "Diversidad Intra-Lista por Método"),
    ]):
        ax = fig.add_subplot(gs[ax_idx])
        for i, (cond, color) in enumerate(zip(conditions, colors)):
            ax.bar(x + i * width, metric_data[cond], width, label=cond,
                   color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Perfil de Usuario Simulado")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        short_labels = [lb.split("—")[0].strip() for lb in user_labels]
        ax.set_xticklabels(short_labels, rotation=15, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Ablación del Pipeline de Recomendación — TasteBud", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="E1 — Ablación de Componentes")
    parser.add_argument("--llm", action="store_true", help="Include SISTEMA condition (OpenAI API calls)")
    args = parser.parse_args()

    set_seed(42)
    ensure_results_dir()

    conditions = ["BAYESIANO", "COSENO", "POPULARIDAD", "ALEATORIO"]
    if args.llm:
        conditions = ["SISTEMA"] + conditions

    print(f"\n=== E1 — Ablación | conditions: {conditions} ===\n")

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items with taste vectors\n")

        rows = []
        user_labels = []

        for profile in SIMULATED_USERS:
            user = build_user(profile)
            label = profile["label"]
            user_labels.append(label)
            safe_items = apply_safety_filter(all_items, user)
            print(f"  {label}: {len(safe_items)} safe items after filtering")

            for cond in conditions:
                try:
                    if cond == "SISTEMA":
                        recs = condition_sistema(safe_items, user, restaurant_id, top_n=10)
                    elif cond == "BAYESIANO":
                        recs = condition_bayesiano(safe_items, user, session, top_n=10)
                    elif cond == "COSENO":
                        recs = condition_coseno(safe_items, user, top_n=10)
                    elif cond == "POPULARIDAD":
                        recs = condition_popularidad(safe_items, top_n=10)
                    elif cond == "ALEATORIO":
                        recs = condition_aleatorio(safe_items, seed=42, top_n=10)
                    else:
                        recs = []

                    tas = compute_tas(user.taste_vector, recs)
                    ild = compute_ild(recs)
                    print(f"    {cond:12s}  TAS={tas:.4f}  ILD={ild:.4f}")
                    rows.append({
                        "user_id": profile["id"],
                        "user_label": label,
                        "condition": cond,
                        "tas_at_10": round(tas, 6),
                        "ild_at_10": round(ild, 6),
                    })
                except Exception as exc:
                    print(f"    {cond:12s}  ERROR: {exc}")
                    rows.append({
                        "user_id": profile["id"],
                        "user_label": label,
                        "condition": cond,
                        "tas_at_10": "",
                        "ild_at_10": "",
                    })
            print()

    save_csv(rows, RESULTS_DIR / "exp1_ablation.csv")
    plot_results(rows, conditions, user_labels, FIGURES_DIR / "exp1_ablation.png")
    print("\nE1 complete.")


if __name__ == "__main__":
    main()
