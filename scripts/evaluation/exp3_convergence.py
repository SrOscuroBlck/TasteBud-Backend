"""
E3 — Convergencia del Perfil Bayesiano en Sesiones Sucesivas
=============================================================
Tracks how the Bayesian taste profile converges toward the user's true
preferences over 10 simulated feedback sessions.

Uses Carlos, Valentina, and Andres (strongest dominant axes: spicy, sweet, umami).
Profiles and users are persisted to the real DB (no cleanup — per research preference).

Usage:
    python scripts/evaluation/exp3_convergence.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sqlmodel import select

from helpers import (
    FIGURES_DIR, RESULTS_DIR,
    apply_safety_filter, compute_tas, cosine_similarity,
    ensure_results_dir, get_db_session, get_evaluation_restaurant,
    save_json, set_seed,
)
from user_profiles import SIMULATED_USERS, build_user

from models.bayesian_profile import BayesianTasteProfile
from models.session import FeedbackType
from models.user import TASTE_AXES, User
from services.core.reranking_service import RecommendationContext, RerankingService
from services.learning.bayesian_profile_service import BayesianProfileService


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
N_SESSIONS = 10
TOP_N = 5
LIKE_THRESHOLD = 0.65
DISLIKE_THRESHOLD = 0.35

# Profiles used in E3 (dominant axis must be clear)
E3_PROFILE_LABELS = ["Carlos — Picante/Umami", "Valentina — Dulce/Suave", "Andres — Umami/Graso"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_eval_user(db_session, user: User) -> User:
    """
    Persist the evaluation user to the DB if not already there.
    Uses the email as the unique key.
    """
    existing = db_session.exec(
        select(User).where(User.email == user.email)
    ).first()
    if existing:
        return existing
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _init_population_profile(db_session, user: User) -> BayesianTasteProfile:
    """
    Initialize from population priors (fallback to taste_vector with moderate uncertainty).
    Creates and persists a fresh BayesianTasteProfile.
    """
    # Remove existing profile if present
    existing = db_session.exec(
        select(BayesianTasteProfile).where(BayesianTasteProfile.user_id == user.id)
    ).first()
    if existing:
        return existing

    # Use population-like means (all axes at 0.5, uncertainty 0.4) as starting point
    pop_user = User(
        id=user.id,
        email=user.email,
        taste_vector={k: 0.5 for k in TASTE_AXES},
        taste_uncertainty={k: 0.4 for k in TASTE_AXES},
        cuisine_affinity={},
        allergies=[],
        dietary_rules=[],
    )
    svc = BayesianProfileService()
    profile = svc.create_profile_from_user(db_session, pop_user)
    db_session.add(profile)
    db_session.commit()
    db_session.refresh(profile)
    return profile


def _generate_feedback(item, true_taste: dict) -> FeedbackType:
    """Deterministic synthetic feedback based on cosine similarity."""
    sim = cosine_similarity(true_taste, item.features or {})
    if sim >= LIKE_THRESHOLD:
        return FeedbackType.LIKE
    elif sim < DISLIKE_THRESHOLD:
        return FeedbackType.DISLIKE
    else:
        return FeedbackType.SKIP


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(all_data, output_path):
    profile_styles = [
        ("Carlos — Picante/Umami",  "spicy", "#e74c3c", 0.90),
        ("Valentina — Dulce/Suave", "sweet", "#e91e8c", 0.90),
        ("Andres — Umami/Graso",    "umami", "#2196f3", 0.90),
    ]

    sessions = list(range(1, N_SESSIONS + 1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, axis, color, target in profile_styles:
        if label not in all_data:
            continue
        data = all_data[label]
        e_theta = [d["e_theta"] for d in data]
        tas_vals = [d["tas_5"] for d in data]

        # E[theta] convergence
        axes[0].plot(sessions, e_theta, marker="o", color=color,
                     label=f"{label.split('—')[0].strip()} ({axis})", linewidth=2)
        axes[0].axhline(y=target, color=color, linestyle="--", alpha=0.4, linewidth=1)

        # TAS@5 evolution
        axes[1].plot(sessions, tas_vals, marker="s", color=color,
                     label=label.split("—")[0].strip(), linewidth=2)

    axes[0].set_xlabel("Sesión de Retroalimentación")
    axes[0].set_ylabel("E[θ] — Preferencia Estimada (Eje Dominante)")
    axes[0].set_title("Convergencia del Perfil Bayesiano")
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_xticks(sessions)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Sesión de Retroalimentación")
    axes[1].set_ylabel("TAS@5")
    axes[1].set_title("Calidad de Recomendación por Sesión")
    axes[1].set_ylim(0.3, 1.0)
    axes[1].set_xticks(sessions)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Aprendizaje del Sistema a lo largo de Sesiones de Retroalimentación",
                 fontsize=12, fontweight="bold")
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

    print("\n=== E3 — Convergencia del Perfil Bayesiano ===\n")

    e3_profiles = [p for p in SIMULATED_USERS if p["label"] in E3_PROFILE_LABELS]
    bp_svc = BayesianProfileService()

    all_data = {}

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items\n")

        for profile_dict in e3_profiles:
            label = profile_dict["label"]
            dominant_axis = profile_dict["dominant_axis"]
            print(f"  Processing: {label}  (dominant: {dominant_axis})")

            true_user = build_user(profile_dict)
            safe_items = apply_safety_filter(all_items, true_user)

            # Persist user and initialize profile from population prior
            db_user = _get_or_create_eval_user(session, true_user)
            profile = _init_population_profile(session, db_user)

            session_records = []

            for k in range(1, N_SESSIONS + 1):
                # Refresh profile from DB to get latest state
                session.refresh(profile)

                # Get top-5 recommendations using current profile
                context = RecommendationContext(time_of_day="afternoon")
                reranker = RerankingService(population_stats=None)
                ranked = reranker.rerank(
                    safe_items, true_user, context,
                    top_n=TOP_N, bayesian_profile=profile
                )
                recs = [r.item for r in ranked]

                # Compute TAS@5 before feedback
                tas = compute_tas(true_user.taste_vector, recs)

                # Generate and apply synthetic feedback
                for item in recs:
                    fb_type = _generate_feedback(item, true_user.taste_vector)
                    if fb_type in (FeedbackType.LIKE, FeedbackType.DISLIKE):
                        bp_svc.update_from_feedback(
                            session, profile, item, fb_type, datetime.utcnow()
                        )

                # SQLAlchemy does not track in-place mutations on JSON columns.
                # Reassign the dicts so the ORM marks them dirty before commit.
                from sqlalchemy.orm.attributes import flag_modified
                flag_modified(profile, "alpha_params")
                flag_modified(profile, "beta_params")
                flag_modified(profile, "mean_preferences")
                flag_modified(profile, "uncertainties")

                # Read metrics from in-memory state (already updated by update_cached_statistics)
                e_theta = profile.mean_preferences.get(dominant_axis, 0.5) if dominant_axis else 0.5
                var_theta = profile.uncertainties.get(dominant_axis, 0.5) if dominant_axis else 0.5

                session.commit()
                session.refresh(profile)

                record = {
                    "session": k,
                    "e_theta": round(e_theta, 6),
                    "var_theta": round(float(var_theta) ** 2, 6),
                    "tas_5": round(tas, 6),
                }
                session_records.append(record)
                print(f"    Session {k:2d}  E[θ_{dominant_axis}]={e_theta:.4f}  TAS@5={tas:.4f}")

            all_data[label] = session_records
            print()

    save_json(all_data, RESULTS_DIR / "exp3_convergence.json")
    plot_convergence(all_data, FIGURES_DIR / "exp3_convergence.png")
    print("E3 complete.")


if __name__ == "__main__":
    main()
