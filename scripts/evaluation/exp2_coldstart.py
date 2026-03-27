"""
E2 — Cold-Start: Comparación de Inicialización del Perfil
==========================================================
Measures TAS@10 on the very first recommendation depending on how the
user profile is initialized before any feedback.

Three initialization methods:
  A) Onboarding  — Simulate 7 fallback "Would You Rather?" questions.
  B) Poblacional — Use PopulationStats priors (or hardcoded fallback).
  C) Plano       — Beta(1,1) uniform prior (maximum uncertainty).

Usage:
    python scripts/evaluation/exp2_coldstart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlmodel import select

from helpers import (
    FIGURES_DIR, RESULTS_DIR,
    apply_safety_filter, clamp01, compute_tas,
    ensure_results_dir, get_db_session, get_evaluation_restaurant,
    save_csv, set_seed,
)
from user_profiles import SIMULATED_USERS, build_user

from config.settings import settings
from models import PopulationStats
from models.bayesian_profile import BayesianTasteProfile
from models.user import TASTE_AXES, User
from services.core.reranking_service import RecommendationContext, RerankingService
from services.learning.bayesian_profile_service import BayesianProfileService
from services.user.onboarding_service import FALLBACK_QUESTIONS


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded population priors (fallback when PopulationStats is absent)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_POPULATION_MEANS = {
    "sweet": 0.55, "sour": 0.40, "salty": 0.50,
    "bitter": 0.35, "umami": 0.60, "fatty": 0.50, "spicy": 0.45,
}
DEFAULT_POPULATION_SIGMA = {k: 0.40 for k in TASTE_AXES}


# ─────────────────────────────────────────────────────────────────────────────
# Initialization helpers
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_onboarding(user: User) -> User:
    """
    Replicate OnboardingService.answer() logic using FALLBACK_QUESTIONS,
    choosing the option whose axis_impacts dot-product is closest to the
    user's true taste_vector (deterministic, reproducible).
    Returns a copy-like User with updated taste_vector and taste_uncertainty.
    """
    K = settings.ONBOARDING_K
    SIGMA_STEP = settings.ONBOARDING_SIGMA_STEP
    UNCHOSEN_DAMPEN = 0.3

    taste = {k: 0.5 for k in TASTE_AXES}
    uncertainty = {k: 0.5 for k in TASTE_AXES}

    for q in FALLBACK_QUESTIONS:
        options = q["options"]
        # Choose option whose impacts align best with user's true taste
        scores = []
        for opt in options:
            impacts = opt.get("axis_impacts", {})
            score = sum(user.taste_vector.get(ax, 0.5) * abs(delta)
                        for ax, delta in impacts.items())
            # Prefer options with positive impact on axes the user scores high
            positive_alignment = sum(
                user.taste_vector.get(ax, 0.5) * delta
                for ax, delta in impacts.items()
                if delta > 0
            )
            scores.append(positive_alignment)

        chosen_idx = int(np.argmax(scores))
        unchosen_idx = 1 - chosen_idx

        chosen_opt = options[chosen_idx]
        unchosen_opt = options[unchosen_idx]

        # Apply chosen impacts
        touched_axes = set()
        for ax, delta in chosen_opt.get("axis_impacts", {}).items():
            if ax in TASTE_AXES:
                taste[ax] = clamp01(taste[ax] + K * delta)
                uncertainty[ax] = max(0.0, uncertainty[ax] - SIGMA_STEP)
                touched_axes.add(ax)

        # Dampen unchosen impacts (only for axes not already touched)
        for ax, delta in unchosen_opt.get("axis_impacts", {}).items():
            if ax in TASTE_AXES and ax not in touched_axes:
                taste[ax] = clamp01(taste[ax] - K * UNCHOSEN_DAMPEN * abs(delta))
                uncertainty[ax] = max(0.0, uncertainty[ax] - SIGMA_STEP)

    user_copy = User(
        id=user.id,
        email=user.email,
        taste_vector=taste,
        taste_uncertainty=uncertainty,
        cuisine_affinity=user.cuisine_affinity,
        allergies=user.allergies,
        dietary_rules=user.dietary_rules,
        onboarding_completed=True,
    )
    return user_copy


def _population_user(user: User, db_session) -> User:
    """Initialize taste_vector from PopulationStats or hardcoded fallback."""
    stats = db_session.exec(select(PopulationStats)).first()
    if stats and stats.axis_prior_mean:
        means = stats.axis_prior_mean
        sigmas = stats.axis_prior_sigma or DEFAULT_POPULATION_SIGMA
    else:
        means = DEFAULT_POPULATION_MEANS
        sigmas = DEFAULT_POPULATION_SIGMA

    return User(
        id=user.id,
        email=user.email,
        taste_vector={k: means.get(k, 0.5) for k in TASTE_AXES},
        taste_uncertainty={k: sigmas.get(k, 0.4) for k in TASTE_AXES},
        cuisine_affinity={},
        allergies=user.allergies,
        dietary_rules=user.dietary_rules,
        onboarding_completed=False,
    )


def _flat_prior_profile(user: User) -> BayesianTasteProfile:
    """Beta(1,1) profile — maximum entropy, no information about preferences."""
    profile = BayesianTasteProfile(user_id=user.id)
    for axis in TASTE_AXES:
        profile.alpha_params[axis] = 1.0
        profile.beta_params[axis] = 1.0
    profile.update_cached_statistics()
    return profile


def _recommend_with_profile(safe_items, user, profile, top_n=10):
    context = RecommendationContext(time_of_day="afternoon")
    reranker = RerankingService(population_stats=None)
    ranked = reranker.rerank(safe_items, user, context, top_n=top_n, bayesian_profile=profile)
    return [r.item for r in ranked]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(rows, user_labels, output_path):
    methods = ["onboarding", "poblacional", "plano"]
    method_labels = ["Onboarding Bayesiano", "Prior Poblacional", "Prior Plano Beta(1,1)"]
    colors = ["#1a9641", "#fdae61", "#d7191c"]

    tas_data = {m: [] for m in methods}
    for label in user_labels:
        for m in methods:
            match = [r for r in rows if r["user_label"] == label and r["method"] == m]
            tas_data[m].append(float(match[0]["tas_at_10"]) if match else 0.0)

    x = np.arange(len(user_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (m, color, ml) in enumerate(zip(methods, colors, method_labels)):
        ax.bar(x + i * width, tas_data[m], width, label=ml, color=color,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Perfil de Usuario Simulado")
    ax.set_ylabel("TAS@10 en Sesión 0")
    ax.set_title("Calidad de Recomendación en Arranque en Frío\nsegún Método de Inicialización del Perfil")
    ax.set_xticks(x + width)
    short_labels = [lb.split("—")[0].strip() for lb in user_labels]
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    ensure_results_dir()

    print("\n=== E2 — Cold-Start: Inicialización del Perfil ===\n")

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items\n")

        bp_svc = BayesianProfileService()
        rows = []
        user_labels = []

        for profile in SIMULATED_USERS:
            true_user = build_user(profile)
            label = profile["label"]
            user_labels.append(label)
            safe_items = apply_safety_filter(all_items, true_user)
            print(f"  {label}: {len(safe_items)} safe items")

            # ── Method A: Onboarding simulation ──────────────────────────────
            try:
                onboarded_user = _simulate_onboarding(true_user)
                onboard_profile = bp_svc.create_profile_from_user(session, onboarded_user)
                recs_a = _recommend_with_profile(safe_items, onboarded_user, onboard_profile)
                tas_a = compute_tas(true_user.taste_vector, recs_a)
            except Exception as e:
                print(f"    onboarding ERROR: {e}")
                tas_a = ""
            print(f"    onboarding   TAS={tas_a:.4f}" if tas_a != "" else "    onboarding   FAILED")

            # ── Method B: Population priors ───────────────────────────────────
            try:
                pop_user = _population_user(true_user, session)
                pop_profile = bp_svc.create_profile_from_user(session, pop_user)
                recs_b = _recommend_with_profile(safe_items, pop_user, pop_profile)
                tas_b = compute_tas(true_user.taste_vector, recs_b)
            except Exception as e:
                print(f"    poblacional  ERROR: {e}")
                tas_b = ""
            print(f"    poblacional  TAS={tas_b:.4f}" if tas_b != "" else "    poblacional  FAILED")

            # ── Method C: Flat Beta(1,1) ──────────────────────────────────────
            try:
                flat_profile = _flat_prior_profile(true_user)
                recs_c = _recommend_with_profile(safe_items, true_user, flat_profile)
                tas_c = compute_tas(true_user.taste_vector, recs_c)
            except Exception as e:
                print(f"    plano        ERROR: {e}")
                tas_c = ""
            print(f"    plano        TAS={tas_c:.4f}" if tas_c != "" else "    plano        FAILED")
            print()

            for method, tas in [("onboarding", tas_a), ("poblacional", tas_b), ("plano", tas_c)]:
                rows.append({
                    "user_id": profile["id"],
                    "user_label": label,
                    "method": method,
                    "tas_at_10": round(tas, 6) if isinstance(tas, float) else "",
                })

    save_csv(rows, RESULTS_DIR / "exp2_coldstart.csv")
    plot_results(rows, user_labels, FIGURES_DIR / "exp2_coldstart.png")
    print("E2 complete.")


if __name__ == "__main__":
    main()
