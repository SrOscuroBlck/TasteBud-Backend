"""
Shared helpers for evaluation experiments:
- DB access
- Metric computation (TAS, ILD, Jaccard)
- Safety filtering
- Seed management and I/O utilities
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from sqlmodel import Session, select

# ── Project root on sys.path so services import cleanly ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.database import engine
from models.restaurant import MenuItem, Restaurant
from models.user import User, TASTE_AXES

# Import directly from the module file to avoid services/features/__init__.py
# which eagerly imports FAISSService (requires the `faiss` native library).
import importlib.util as _ilu
import os as _os
_features_path = _os.path.join(_PROJECT_ROOT, "services", "features", "features.py")
_spec = _ilu.spec_from_file_location("features_module", _features_path)
_features_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_features_mod)
cosine_similarity = _features_mod.cosine_similarity
has_allergen = _features_mod.has_allergen
violates_diet = _features_mod.violates_diet
clamp01 = _features_mod.clamp01

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "Tesis" / "USB_template" / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db_session() -> Session:
    """Return a new SQLModel Session using the configured engine."""
    return Session(engine)


def get_evaluation_restaurant(session: Session) -> Tuple[UUID, List[MenuItem]]:
    """
    Return (restaurant_id, items) for the first restaurant that has >= 20
    menu items with non-empty `features` dicts.
    Exits with a clear error message if no suitable restaurant is found.
    """
    restaurants = session.exec(select(Restaurant)).all()
    if not restaurants:
        sys.exit(
            "ERROR: No restaurants found in the database.\n"
            "Run the menu ingestion pipeline first."
        )

    for restaurant in restaurants:
        items = session.exec(
            select(MenuItem).where(MenuItem.restaurant_id == restaurant.id)
        ).all()
        featured = [it for it in items if it.features]
        if len(featured) >= 20:
            return restaurant.id, featured

    sys.exit(
        "ERROR: No restaurant with >= 20 items that have taste vectors (features) found.\n"
        "Run: python scripts/maintenance/regenerate_taste_vectors_llm.py"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tas(user_taste: Dict[str, float], items: List[MenuItem]) -> float:
    """
    Taste Alignment Score: cosine similarity between the user's taste vector
    and the mean feature vector of the given items.
    Returns 0.0 if items is empty or all features are zero.
    """
    if not items:
        return 0.0
    mean_vec: Dict[str, float] = {axis: 0.0 for axis in TASTE_AXES}
    count = 0
    for item in items:
        if not item.features:
            continue
        for axis in TASTE_AXES:
            mean_vec[axis] += item.features.get(axis, 0.0)
        count += 1
    if count == 0:
        return 0.0
    for axis in TASTE_AXES:
        mean_vec[axis] /= count
    return cosine_similarity(user_taste, mean_vec)


def compute_ild(items: List[MenuItem]) -> float:
    """
    Intra-List Diversity: average pairwise (1 - cosine_similarity) across all
    unique item pairs. Returns 0.0 when fewer than 2 items.
    """
    valid = [it for it in items if it.features]
    if len(valid) < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            sim = cosine_similarity(valid[i].features, valid[j].features)
            total += 1.0 - sim
            count += 1
    return total / count if count > 0 else 0.0


def jaccard_distance(list_a: List, list_b: List) -> float:
    """1 - Jaccard similarity between two lists (converted to sets of strings)."""
    set_a = {str(x) for x in list_a}
    set_b = {str(x) for x in list_b}
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - intersection / union


def profile_distance(user_a: User, user_b: User) -> float:
    """1 - cosine_similarity between the taste vectors of two users."""
    return 1.0 - cosine_similarity(user_a.taste_vector, user_b.taste_vector)


# ─────────────────────────────────────────────────────────────────────────────
# Safety filtering
# ─────────────────────────────────────────────────────────────────────────────

def apply_safety_filter(items: List[MenuItem], user: User) -> List[MenuItem]:
    """
    Remove items that violate the user's allergen constraints or dietary rules.
    Uses the canonical has_allergen() and violates_diet() functions.
    """
    safe = []
    for item in items:
        if has_allergen(user.allergies, item.ingredients or [], item.allergens or []):
            continue
        if violates_diet(user.dietary_rules, item.dietary_tags or []):
            continue
        safe.append(item)
    return safe


def verify_no_violations(filtered_items: List[MenuItem], user: User) -> int:
    """
    Check that no filtered item violates user constraints.
    Returns the count of violations (should always be 0).
    """
    violations = 0
    for item in filtered_items:
        if has_allergen(user.allergies, item.ingredients or [], item.allergens or []):
            violations += 1
        elif violates_diet(user.dietary_rules, item.dietary_tags or []):
            violations += 1
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV → {path}")


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON → {path}")
