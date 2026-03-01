from __future__ import annotations
import os
import re

from models.user import TASTE_AXES
from services.features.features import (
    CANON_INGREDIENTS,
    apply_tag_modifiers,
    cosine_similarity,
    generate_keyword_matching_profile,
)
from services.user.onboarding_service import FALLBACK_QUESTIONS

CANONICAL_SET = set(TASTE_AXES)


class TestCanonIngredientsKeys:
    def test_all_axes_use_taste_axes(self):
        for ingredient, data in CANON_INGREDIENTS.items():
            for axis_key in data.get("axes", {}):
                assert axis_key in CANONICAL_SET, (
                    f"CANON_INGREDIENTS['{ingredient}'] uses non-canonical axis '{axis_key}'. "
                    f"Valid axes: {TASTE_AXES}"
                )


class TestTagModifiers:
    def test_apply_tag_modifiers_uses_canonical_keys(self):
        profile = {k: 0.0 for k in TASTE_AXES}
        test_tags = [
            "fried", "grilled", "spicy", "sweet", "creamy", "smoked",
            "tangy", "crispy", "rich", "herb", "citrus", "fermented",
        ]
        result = apply_tag_modifiers(profile, test_tags)
        for key in result:
            assert key in CANONICAL_SET, (
                f"apply_tag_modifiers returned non-canonical key: '{key}'"
            )

    def test_no_stale_keys_in_output(self):
        stale_keys = {"fattiness", "acidity", "crunch", "temp_hot"}
        profile = {k: 0.0 for k in TASTE_AXES}
        all_tags = [
            "fried", "grilled", "baked", "steamed", "raw", "smoked",
            "spicy", "sweet", "sour", "bitter", "salty", "savory",
            "creamy", "crispy", "tangy", "rich",
        ]
        result = apply_tag_modifiers(profile, all_tags)
        for key in result:
            assert key not in stale_keys, (
                f"apply_tag_modifiers output contains stale key: '{key}'"
            )


class TestKeywordMatchingProfile:
    def test_generated_profiles_only_use_canonical_keys(self):
        test_items = [
            ("Spicy Chicken Wings", ["chicken", "hot_sauce", "butter"], ["spicy", "fried"]),
            ("Caesar Salad", ["lettuce", "parmesan", "croutons"], ["fresh", "savory"]),
            ("Chocolate Cake", ["chocolate", "butter", "sugar"], ["sweet", "rich"]),
            ("Pad Thai", ["noodles", "shrimp", "peanut"], ["sour", "umami"]),
            ("Margherita Pizza", ["dough", "tomato", "mozzarella"], ["cheesy", "baked"]),
        ]
        for name, ingredients, tags in test_items:
            profile = generate_keyword_matching_profile(ingredients, tags, name, "")
            for key in profile:
                assert key in CANONICAL_SET, (
                    f"generate_keyword_matching_profile('{name}') returned "
                    f"non-canonical key: '{key}'"
                )


class TestCosineWithMatchedKeys:
    def test_identical_vectors_score_one(self):
        v = {k: 0.5 for k in TASTE_AXES}
        assert abs(cosine_similarity(v, v) - 1.0) < 0.01

    def test_orthogonal_vectors_score_zero(self):
        a = {"sweet": 1.0, "sour": 0.0, "salty": 0.0, "bitter": 0.0, "umami": 0.0, "fatty": 0.0, "spicy": 0.0}
        b = {"sweet": 0.0, "sour": 1.0, "salty": 0.0, "bitter": 0.0, "umami": 0.0, "fatty": 0.0, "spicy": 0.0}
        sim = cosine_similarity(a, b)
        assert sim < 0.01, f"Orthogonal vectors should score ~0, got {sim}"

    def test_mismatched_keys_contribute_nothing(self):
        user_vec = {"sweet": 0.8, "spicy": 0.9, "salty": 0.7}
        item_with_correct_keys = {"sweet": 0.8, "spicy": 0.9, "salty": 0.7}
        item_with_wrong_keys = {"fattiness": 0.8, "acidity": 0.9, "crunch": 0.7}

        sim_correct = cosine_similarity(user_vec, item_with_correct_keys)
        sim_wrong = cosine_similarity(user_vec, item_with_wrong_keys)

        assert sim_correct > 0.9, f"Matched keys should give high similarity, got {sim_correct}"
        assert sim_wrong < 0.01, (
            f"Mismatched keys should contribute zero similarity, got {sim_wrong}. "
            f"This means stale keys like 'fattiness' are still present somewhere."
        )


class TestOnboardingFallbackKeysConsistency:
    def test_all_fallback_axis_impacts_are_canonical(self):
        for i, q in enumerate(FALLBACK_QUESTIONS):
            for opt in q["options"]:
                for key in opt.get("axis_impacts", {}):
                    assert key in CANONICAL_SET, (
                        f"Fallback question {i}, option '{opt['label']}' "
                        f"has non-canonical axis_impacts key: '{key}'"
                    )


class TestNoStaleKeysInSourceFiles:
    STALE_PATTERNS = [
        r'"fattiness"',
        r"'fattiness'",
        r'"acidity"',
        r"'acidity'",
        r'"crunch"',
        r"'crunch'",
        r'"temp_hot"',
        r"'temp_hot'",
    ]

    CRITICAL_FILES = [
        "services/features/features.py",
        "services/user/onboarding_service.py",
        "services/core/retrieval_service.py",
        "data/seed.py",
        "services/learning/unified_feedback_service.py",
        "services/learning/in_session_learning_service.py",
    ]

    def test_no_stale_string_literals_in_critical_files(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        violations = []

        for rel_path in self.CRITICAL_FILES:
            full_path = os.path.join(base, rel_path)
            if not os.path.exists(full_path):
                continue
            with open(full_path) as f:
                content = f.read()
            for pattern in self.STALE_PATTERNS:
                matches = re.findall(pattern, content)
                if matches:
                    violations.append(f"{rel_path}: found {pattern} ({len(matches)}x)")

        assert not violations, (
            "Stale feature keys found in source files:\n" + "\n".join(violations)
        )
