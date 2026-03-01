from __future__ import annotations
from types import SimpleNamespace
from uuid import uuid4

from models.user import TASTE_AXES
from services.learning.unified_feedback_service import LEARNING_RATE_MAP
from services.learning.in_session_learning_service import InSessionLearningService


def _make_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        taste_vector={k: 0.5 for k in TASTE_AXES},
        taste_uncertainty={k: 0.3 for k in TASTE_AXES},
        cuisine_affinity={"Mexican": 0.5},
        liked_ingredients=[],
        disliked_ingredients=[],
        allergies=[],
        dietary_rules=[],
        ingredient_penalties={},
        onboarding_choices=[],
    )


def _make_spicy_item() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name="Spicy Buffalo Wings",
        description="Fiery hot wings",
        ingredients=["chicken", "hot_sauce", "butter"],
        cuisine=["American"],
        course="appetizer",
        price=12.0,
        features={
            "spicy": 0.9,
            "fatty": 0.6,
            "salty": 0.7,
            "umami": 0.5,
            "sweet": 0.1,
            "sour": 0.2,
            "bitter": 0.1,
        },
    )


def _make_sweet_item() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        name="Chocolate Lava Cake",
        description="Rich chocolate dessert",
        ingredients=["chocolate", "butter", "sugar"],
        cuisine=["French"],
        course="dessert",
        price=10.0,
        features={
            "sweet": 0.9,
            "fatty": 0.7,
            "bitter": 0.3,
            "umami": 0.1,
            "spicy": 0.0,
            "salty": 0.1,
            "sour": 0.0,
        },
    )


def _make_feedback(item_id, feedback_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        item_id=item_id,
        feedback_type=feedback_type,
    )


def _make_session() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
    )


class TestLearningRatesAreStrong:
    def test_learning_rates_are_meaningful(self):
        assert LEARNING_RATE_MAP["mild"] >= 0.05, (
            f"mild learning rate {LEARNING_RATE_MAP['mild']} too weak"
        )
        assert LEARNING_RATE_MAP["medium"] >= 0.10, (
            f"medium learning rate {LEARNING_RATE_MAP['medium']} too weak"
        )
        assert LEARNING_RATE_MAP["strong"] >= 0.20, (
            f"strong learning rate {LEARNING_RATE_MAP['strong']} too weak"
        )

    def test_strong_dislike_produces_detectable_shift(self):
        learning_rate = LEARNING_RATE_MAP["strong"]
        item = _make_spicy_item()

        shifted_axes = 0
        for axis, value in item.features.items():
            if value > 0.5:
                delta = learning_rate * value * 2.0
                if delta > 0.05:
                    shifted_axes += 1

        assert shifted_axes >= 2, (
            f"Only {shifted_axes} axes would shift meaningfully from a strong dislike. "
            f"Learning rates are still too weak."
        )


class TestInSessionLearningAdjustments:
    def test_calculate_session_adjustments_not_all_zeros(self):
        service = InSessionLearningService()
        item = _make_spicy_item()
        user = _make_user()
        session = _make_session()
        feedback = _make_feedback(item.id, "like")
        items_map = {str(item.id): item}

        adjustments = service.calculate_session_adjustments(
            user, [feedback], session, items_map
        )

        non_zero = {k: v for k, v in adjustments.items() if v != 0.0}
        assert len(non_zero) > 0, (
            "calculate_session_adjustments returned all zeros even with a 'like' feedback"
        )

    def test_like_boosts_matching_axes(self):
        service = InSessionLearningService()
        item = _make_spicy_item()
        user = _make_user()
        session = _make_session()
        feedback = _make_feedback(item.id, "like")
        items_map = {str(item.id): item}

        adjustments = service.calculate_session_adjustments(
            user, [feedback], session, items_map
        )

        assert adjustments.get("spicy", 0.0) > 0, (
            f"Liking a spicy item should boost 'spicy' axis, got {adjustments.get('spicy', 0.0)}"
        )

    def test_dislike_reduces_matching_axes(self):
        service = InSessionLearningService()
        item = _make_spicy_item()
        user = _make_user()
        session = _make_session()
        feedback = _make_feedback(item.id, "dislike")
        items_map = {str(item.id): item}

        adjustments = service.calculate_session_adjustments(
            user, [feedback], session, items_map
        )

        assert adjustments.get("spicy", 0.0) < 0, (
            f"Disliking a spicy item should reduce 'spicy' axis, got {adjustments.get('spicy', 0.0)}"
        )

    def test_adjustments_use_canonical_keys_only(self):
        service = InSessionLearningService()
        item = _make_spicy_item()
        user = _make_user()
        session = _make_session()
        feedback = _make_feedback(item.id, "like")
        items_map = {str(item.id): item}

        adjustments = service.calculate_session_adjustments(
            user, [feedback], session, items_map
        )

        canonical = set(TASTE_AXES)
        for key in adjustments:
            assert key in canonical, (
                f"Session adjustment returned non-canonical key: '{key}'"
            )

    def test_multiple_dislikes_compound(self):
        service = InSessionLearningService()
        spicy1 = _make_spicy_item()
        spicy2 = _make_spicy_item()
        spicy2.name = "Hot Jalapeño Poppers"

        user = _make_user()
        session = _make_session()
        fb1 = _make_feedback(spicy1.id, "dislike")
        fb2 = _make_feedback(spicy2.id, "dislike")

        items_map = {str(spicy1.id): spicy1, str(spicy2.id): spicy2}
        adjustments = service.calculate_session_adjustments(
            user, [fb1, fb2], session, items_map
        )

        assert adjustments.get("spicy", 0.0) < -0.1, (
            f"Two spicy dislikes should produce adjustment < -0.1, "
            f"got {adjustments.get('spicy', 0.0)}"
        )


class TestApplyImmediateLearning:
    def test_dislike_lowers_high_feature_axes(self):
        service = InSessionLearningService()
        original_tv = {k: 0.5 for k in TASTE_AXES}
        item = _make_spicy_item()

        updated = service.apply_immediate_learning(
            original_tv, item.features, "dislike", weight=0.15
        )

        assert updated["spicy"] < original_tv["spicy"], (
            "Disliking a spicy item should lower the user's spicy score"
        )

    def test_like_raises_high_feature_axes(self):
        service = InSessionLearningService()
        original_tv = {k: 0.5 for k in TASTE_AXES}
        item = _make_sweet_item()

        updated = service.apply_immediate_learning(
            original_tv, item.features, "like", weight=0.15
        )

        assert updated["sweet"] > original_tv["sweet"], (
            "Liking a sweet item should raise the user's sweet score"
        )

    def test_output_only_contains_canonical_keys(self):
        service = InSessionLearningService()
        original_tv = {k: 0.5 for k in TASTE_AXES}
        item_features = {"spicy": 0.9, "fattiness": 0.8, "crunch": 0.7}

        updated = service.apply_immediate_learning(
            original_tv, item_features, "like", weight=0.1
        )

        canonical = set(TASTE_AXES)
        for key in updated:
            assert key in canonical, (
                f"apply_immediate_learning returned non-canonical key: '{key}'"
            )
