from __future__ import annotations
import copy
from models.user import TASTE_AXES, User
from services.features.features import clamp01
from config.settings import settings
from services.user.onboarding_service import FALLBACK_QUESTIONS


def _make_fresh_user() -> dict:
    return {
        "taste_vector": {k: 0.5 for k in TASTE_AXES},
        "taste_uncertainty": {k: 0.5 for k in TASTE_AXES},
    }


def _apply_choice(tv: dict, tu: dict, option: dict) -> None:
    axis_impacts = option.get("axis_impacts", {})
    touched = set()
    for axis, delta in axis_impacts.items():
        if axis not in TASTE_AXES:
            continue
        old = tv.get(axis, 0.5)
        tv[axis] = clamp01(old + settings.ONBOARDING_K * float(delta))
        touched.add(axis)
    for axis in touched:
        tu[axis] = max(0.0, tu.get(axis, 0.5) - settings.ONBOARDING_SIGMA_STEP)


class TestFallbackQuestionsUseCanonicalKeys:
    def test_all_axis_impacts_use_taste_axes(self):
        for i, q in enumerate(FALLBACK_QUESTIONS):
            for opt in q["options"]:
                for key in opt["axis_impacts"]:
                    assert key in TASTE_AXES, (
                        f"Question {i} option '{opt['label']}' has non-canonical axis key: '{key}'"
                    )

    def test_every_question_has_two_options_with_axis_impacts(self):
        for i, q in enumerate(FALLBACK_QUESTIONS):
            assert len(q["options"]) == 2, f"Question {i} should have 2 options"
            for opt in q["options"]:
                assert "axis_impacts" in opt, (
                    f"Question {i} option '{opt['label']}' missing axis_impacts"
                )
                assert len(opt["axis_impacts"]) > 0, (
                    f"Question {i} option '{opt['label']}' has empty axis_impacts"
                )

    def test_no_axis_hints_key_remains(self):
        for i, q in enumerate(FALLBACK_QUESTIONS):
            assert "axis_hints" not in q, (
                f"Question {i} still has legacy 'axis_hints' key — should be per-option axis_impacts"
            )


class TestOnboardingDivergence:
    def test_opposite_choices_produce_divergent_profiles(self):
        user_a = _make_fresh_user()
        user_b = _make_fresh_user()

        for q in FALLBACK_QUESTIONS:
            option_a = q["options"][0]
            option_b = q["options"][1]
            _apply_choice(user_a["taste_vector"], user_a["taste_uncertainty"], option_a)
            _apply_choice(user_b["taste_vector"], user_b["taste_uncertainty"], option_b)

        diff = sum(
            abs(user_a["taste_vector"][k] - user_b["taste_vector"][k]) for k in TASTE_AXES
        )
        per_axis_avg_diff = diff / len(TASTE_AXES)

        assert diff > 1.0, (
            f"Total taste vector divergence is only {diff:.3f} after 7 opposite choices — "
            f"should be > 1.0. User A: {user_a['taste_vector']}, User B: {user_b['taste_vector']}"
        )
        assert per_axis_avg_diff > 0.10, (
            f"Average per-axis divergence is only {per_axis_avg_diff:.3f} — too small to matter"
        )

    def test_single_choice_produces_measurable_shift(self):
        tv = {k: 0.5 for k in TASTE_AXES}
        tu = {k: 0.5 for k in TASTE_AXES}
        option = FALLBACK_QUESTIONS[0]["options"][0]

        _apply_choice(tv, tu, option)

        shifted_axes = [k for k in TASTE_AXES if abs(tv[k] - 0.5) > 0.01]
        assert len(shifted_axes) >= 2, (
            f"Only {len(shifted_axes)} axes shifted after answering first question. "
            f"Expected at least 2. Vector: {tv}"
        )

        max_shift = max(abs(tv[k] - 0.5) for k in TASTE_AXES)
        assert max_shift >= 0.05, (
            f"Max shift is only {max_shift:.3f} — with K={settings.ONBOARDING_K} this is too small. "
            f"Vector: {tv}"
        )

    def test_uncertainty_decreases_on_touched_axes(self):
        tv = {k: 0.5 for k in TASTE_AXES}
        tu = {k: 0.5 for k in TASTE_AXES}
        option = FALLBACK_QUESTIONS[0]["options"][0]
        touched_axes = set(option["axis_impacts"].keys())

        _apply_choice(tv, tu, option)

        for axis in touched_axes:
            assert tu[axis] < 0.5, (
                f"Uncertainty on '{axis}' did not decrease after being touched"
            )
        untouched_axes = set(TASTE_AXES) - touched_axes
        for axis in untouched_axes:
            assert tu[axis] == 0.5, (
                f"Uncertainty on untouched axis '{axis}' changed unexpectedly"
            )

    def test_bold_user_vs_mild_user(self):
        bold_tv = {k: 0.5 for k in TASTE_AXES}
        bold_tu = {k: 0.5 for k in TASTE_AXES}
        mild_tv = {k: 0.5 for k in TASTE_AXES}
        mild_tu = {k: 0.5 for k in TASTE_AXES}

        bold_choices = [0, 1, 0, 0, 0, 0, 0]
        mild_choices = [1, 1, 1, 1, 1, 1, 1]

        for i, q in enumerate(FALLBACK_QUESTIONS):
            bold_opt = q["options"][bold_choices[i]]
            mild_opt = q["options"][mild_choices[i]]
            _apply_choice(bold_tv, bold_tu, bold_opt)
            _apply_choice(mild_tv, mild_tu, mild_opt)

        assert bold_tv["spicy"] > mild_tv["spicy"], (
            "Bold user should score higher on spicy"
        )
        assert bold_tv["fatty"] > mild_tv["fatty"], (
            "Bold user should score higher on fatty"
        )

    def test_all_seven_questions_keep_vector_in_bounds(self):
        tv = {k: 0.5 for k in TASTE_AXES}
        tu = {k: 0.5 for k in TASTE_AXES}

        for q in FALLBACK_QUESTIONS:
            _apply_choice(tv, tu, q["options"][0])

        for axis in TASTE_AXES:
            assert 0.0 <= tv[axis] <= 1.0, f"taste_vector[{axis}] = {tv[axis]} out of bounds"
            assert 0.0 <= tu[axis] <= 0.5, f"uncertainty[{axis}] = {tu[axis]} out of bounds"


class TestSettingsConsistency:
    def test_onboarding_k_is_strong_enough(self):
        assert settings.ONBOARDING_K >= 0.3, (
            f"ONBOARDING_K={settings.ONBOARDING_K} is too low for meaningful learning. "
            f"Should be >= 0.3"
        )

    def test_sigma_step_matches_question_count(self):
        initial_sigma = 0.5
        after_7 = initial_sigma - (7 * settings.ONBOARDING_SIGMA_STEP)
        assert after_7 <= 0.1, (
            f"After 7 questions, uncertainty would be {after_7:.2f} — "
            f"still too high (should be <= 0.1) with SIGMA_STEP={settings.ONBOARDING_SIGMA_STEP}"
        )
