"""
Synthetic user profiles for evaluation experiments.
Each profile represents a distinct region of the taste space.
"""
from uuid import uuid4
from models.user import User, TASTE_AXES


SIMULATED_USERS = [
    {
        "id": str(uuid4()),
        "label": "Carlos — Picante/Umami",
        "taste_vector": {
            "spicy": 0.95, "umami": 0.75, "fatty": 0.65, "salty": 0.65,
            "sour": 0.35, "sweet": 0.10, "bitter": 0.25,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"Mexican": 0.90, "Thai": 0.85, "Indian": 0.80},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "spicy",
        "target_dominant_value": 0.90,
    },
    {
        "id": str(uuid4()),
        "label": "Valentina — Dulce/Suave",
        "taste_vector": {
            "sweet": 0.92, "fatty": 0.55, "sour": 0.35, "salty": 0.22,
            "umami": 0.18, "spicy": 0.05, "bitter": 0.12,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"French": 0.70},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "sweet",
        "target_dominant_value": 0.90,
    },
    {
        "id": str(uuid4()),
        "label": "Andres — Umami/Graso",
        "taste_vector": {
            "umami": 0.92, "fatty": 0.80, "salty": 0.70, "bitter": 0.38,
            "spicy": 0.42, "sour": 0.28, "sweet": 0.18,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"Japanese": 0.85, "French": 0.80, "Italian": 0.75},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "umami",
        "target_dominant_value": 0.90,
    },
    {
        "id": str(uuid4()),
        "label": "Maria — Vegana/Acida",
        "taste_vector": {
            "sour": 0.85, "sweet": 0.48, "bitter": 0.45, "spicy": 0.38,
            "salty": 0.40, "umami": 0.28, "fatty": 0.18,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"Mediterranean": 0.80},
        "allergies": [],
        "dietary_rules": ["vegetarian", "vegan"],
        "dominant_axis": "sour",
        "target_dominant_value": 0.85,
    },
    {
        "id": str(uuid4()),
        "label": "Santiago — Equilibrado/Explorador",
        "taste_vector": {
            "umami": 0.58, "salty": 0.55, "fatty": 0.50, "sweet": 0.45,
            "sour": 0.42, "spicy": 0.38, "bitter": 0.32,
        },
        "taste_uncertainty": {k: 0.40 for k in TASTE_AXES},
        "cuisine_affinity": {},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": None,
        "target_dominant_value": None,
    },
    {
        "id": str(uuid4()),
        "label": "Isabella — Sin Gluten",
        "taste_vector": {
            "sweet": 0.72, "umami": 0.55, "fatty": 0.48, "salty": 0.50,
            "sour": 0.32, "spicy": 0.20, "bitter": 0.18,
        },
        "taste_uncertainty": {k: 0.10 for k in TASTE_AXES},
        "cuisine_affinity": {},
        "allergies": ["gluten", "wheat"],
        "dietary_rules": [],
        "dominant_axis": "sweet",
        "target_dominant_value": 0.72,
    },
    # ── Additional restriction-free profiles (for E4 statistical power) ──────
    {
        "id": str(uuid4()),
        "label": "Marcos — Amargo/Acido",
        "taste_vector": {
            "bitter": 0.88, "sour": 0.72, "salty": 0.45, "umami": 0.35,
            "fatty": 0.28, "spicy": 0.22, "sweet": 0.12,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"Mediterranean": 0.75, "Greek": 0.70},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "bitter",
        "target_dominant_value": 0.88,
    },
    {
        "id": str(uuid4()),
        "label": "Lucia — Graso/Salado",
        "taste_vector": {
            "fatty": 0.90, "salty": 0.82, "umami": 0.65, "bitter": 0.30,
            "spicy": 0.28, "sour": 0.18, "sweet": 0.10,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"American": 0.80, "BBQ": 0.85},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "fatty",
        "target_dominant_value": 0.90,
    },
    {
        "id": str(uuid4()),
        "label": "Diego — Picante/Acido",
        "taste_vector": {
            "spicy": 0.88, "sour": 0.75, "bitter": 0.52, "salty": 0.45,
            "umami": 0.40, "fatty": 0.28, "sweet": 0.08,
        },
        "taste_uncertainty": {k: 0.05 for k in TASTE_AXES},
        "cuisine_affinity": {"Mexican": 0.85, "Korean": 0.80},
        "allergies": [],
        "dietary_rules": [],
        "dominant_axis": "spicy",
        "target_dominant_value": 0.88,
    },
]


def build_user(profile: dict) -> User:
    """Construct an in-memory (detached) User from a profile dict.
    These are NOT persisted to the DB unless explicitly added.
    """
    from uuid import UUID
    return User(
        id=UUID(profile["id"]),
        email=f"eval_{profile['id'][:8]}@tastebud.test",
        taste_vector=profile["taste_vector"],
        taste_uncertainty=profile["taste_uncertainty"],
        cuisine_affinity=profile["cuisine_affinity"],
        allergies=profile["allergies"],
        dietary_rules=profile["dietary_rules"],
        onboarding_completed=True,
    )
