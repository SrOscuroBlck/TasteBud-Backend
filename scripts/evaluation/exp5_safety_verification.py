"""
E5 — Verificación Sistemática del Filtrado de Seguridad
=======================================================
Verifies that the safety filtering pipeline produces zero dietary/allergen
violations across 50 random user variants per dietary condition.

Three conditions:
  - Vegano            (dietary_rules=["vegan"], allergies=[])
  - Sin gluten        (dietary_rules=[], allergies=["gluten"])
  - Vegano+Sin gluten (dietary_rules=["vegan"], allergies=["gluten"])

Usage:
    python scripts/evaluation/exp5_safety_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from helpers import (
    RESULTS_DIR,
    apply_safety_filter, verify_no_violations,
    ensure_results_dir, get_db_session, get_evaluation_restaurant,
    save_csv, set_seed,
)

from models.user import User, TASTE_AXES


# ─────────────────────────────────────────────────────────────────────────────
# Conditions
# ─────────────────────────────────────────────────────────────────────────────

CONDITIONS = [
    {
        "name": "Vegano",
        "dietary_rules": ["vegan"],
        "allergies": [],
    },
    {
        "name": "Sin gluten",
        "dietary_rules": [],
        "allergies": ["gluten"],
    },
    {
        "name": "Vegano + Sin gluten",
        "dietary_rules": ["vegan"],
        "allergies": ["gluten"],
    },
]

N_RUNS = 50


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    ensure_results_dir()

    print("\n=== E5 — Verificación del Filtrado de Seguridad ===\n")
    print(f"Conditions: {[c['name'] for c in CONDITIONS]}")
    print(f"Runs per condition: {N_RUNS}\n")

    rows = []
    summary_rows = []

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items (pool)\n")

        for condition in CONDITIONS:
            cond_name = condition["name"]
            cond_violations = 0
            initial_counts = []
            filtered_counts = []

            print(f"  Condition: {cond_name}")

            for run in range(N_RUNS):
                # Random taste vector (seed is deterministic per run)
                np.random.seed(42 + run)
                taste = {ax: float(v) for ax, v in zip(TASTE_AXES, np.random.uniform(0.1, 0.9, len(TASTE_AXES)))}

                user = User(
                    id=uuid4(),
                    email=f"eval_safety_{run}@tastebud.test",
                    taste_vector=taste,
                    taste_uncertainty={k: 0.3 for k in TASTE_AXES},
                    cuisine_affinity={},
                    allergies=condition["allergies"],
                    dietary_rules=condition["dietary_rules"],
                    onboarding_completed=False,
                )

                n_initial = len(all_items)
                filtered = apply_safety_filter(all_items, user)
                n_filtered = len(filtered)
                violations = verify_no_violations(filtered, user)

                cond_violations += violations
                initial_counts.append(n_initial)
                filtered_counts.append(n_filtered)

                rows.append({
                    "condition": cond_name,
                    "run": run,
                    "items_initial": n_initial,
                    "items_filtered": n_filtered,
                    "reduction_pct": round(100.0 * (n_initial - n_filtered) / max(n_initial, 1), 2),
                    "violations": violations,
                })

            avg_initial = np.mean(initial_counts)
            avg_filtered = np.mean(filtered_counts)
            avg_reduction = 100.0 * (avg_initial - avg_filtered) / max(avg_initial, 1)

            print(f"    Items initial:  {avg_initial:.0f} (constant)")
            print(f"    Items after filter: {avg_filtered:.1f} ± {np.std(filtered_counts):.1f}")
            print(f"    Reduction:      {avg_reduction:.1f}%")
            print(f"    Violations:     {cond_violations} / {N_RUNS} runs")
            adherence = "100%" if cond_violations == 0 else f"{100*(1 - cond_violations/N_RUNS):.1f}%"
            print(f"    Adherence rate: {adherence}\n")

            summary_rows.append({
                "condition": cond_name,
                "n_runs": N_RUNS,
                "items_initial_avg": round(avg_initial, 1),
                "items_filtered_avg": round(avg_filtered, 1),
                "reduction_pct": round(avg_reduction, 1),
                "total_violations": cond_violations,
                "adherence_rate": adherence,
            })

    save_csv(rows, RESULTS_DIR / "exp5_safety.csv")

    # Print summary table
    print("\n" + "=" * 72)
    print("TABLA RESUMEN — Verificación del Filtrado de Seguridad")
    print("=" * 72)
    header = f"{'Condición':<22} | {'N':>4} | {'Items ini':>10} | {'Reducción':>9} | {'Violaciones':>11} | {'Tasa':>6}"
    print(header)
    print("-" * 72)
    for r in summary_rows:
        print(
            f"{r['condition']:<22} | {r['n_runs']:>4} | "
            f"{r['items_initial_avg']:>10.1f} | "
            f"{r['reduction_pct']:>8.1f}% | "
            f"{r['total_violations']:>11} | "
            f"{r['adherence_rate']:>6}"
        )
    print("=" * 72)
    print("\nE5 complete.")


if __name__ == "__main__":
    main()
