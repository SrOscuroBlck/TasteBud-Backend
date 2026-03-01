from __future__ import annotations
import importlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from models.user import TASTE_AXES

_spec = importlib.util.spec_from_file_location(
    "llm_reranking_service",
    "services/ml/llm_reranking_service.py",
)
llm_svc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(llm_svc)

LLMRecommendationError = llm_svc.LLMRecommendationError
_build_user_profile_block = llm_svc._build_user_profile_block
_build_onboarding_block = llm_svc._build_onboarding_block
_build_session_feedback_block = llm_svc._build_session_feedback_block
_build_candidate_block = llm_svc._build_candidate_block
_build_context_block = llm_svc._build_context_block
rerank_single_items = llm_svc.rerank_single_items
compose_full_meal = llm_svc.compose_full_meal


def _make_user(
    spicy_score: float = 0.5,
    sweet_score: float = 0.5,
    onboarding_choices: list | None = None,
) -> SimpleNamespace:
    tv = {k: 0.5 for k in TASTE_AXES}
    tv["spicy"] = spicy_score
    tv["sweet"] = sweet_score
    return SimpleNamespace(
        id=uuid4(),
        taste_vector=tv,
        taste_uncertainty={k: 0.2 for k in TASTE_AXES},
        cuisine_affinity={"Mexican": 0.8, "Italian": 0.6},
        liked_ingredients=["chili", "beef"],
        disliked_ingredients=["cilantro"],
        allergies=["peanut"],
        dietary_rules=[],
        onboarding_choices=onboarding_choices or [],
    )


def _make_item(name: str, course: str = "main", cuisine: list | None = None, spicy: float = 0.5) -> SimpleNamespace:
    feats = {k: 0.3 for k in TASTE_AXES}
    feats["spicy"] = spicy
    return SimpleNamespace(
        id=uuid4(),
        name=name,
        description=f"A delicious {name}",
        ingredients=["chicken", "sauce"],
        cuisine=cuisine or ["American"],
        course=course,
        price=15.0,
        features=feats,
    )


def _make_session(meal_intent: str = "main_only") -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        restaurant_id=uuid4(),
        meal_intent=meal_intent,
        time_of_day="evening",
        detected_hour=19,
        hunger_level="hungry",
        budget=50.0,
        mood="adventurous",
        occasion=None,
        party_size=1,
    )


class TestBuildUserProfileBlock:
    def test_includes_taste_scores(self):
        user = _make_user(spicy_score=0.9, sweet_score=0.2)
        block = _build_user_profile_block(user)
        assert "spicy" in block
        assert "0.90" in block or "0.9" in block

    def test_includes_allergies(self):
        user = _make_user()
        block = _build_user_profile_block(user)
        assert "peanut" in block

    def test_includes_liked_disliked(self):
        user = _make_user()
        block = _build_user_profile_block(user)
        assert "chili" in block
        assert "cilantro" in block


class TestBuildOnboardingBlock:
    def test_empty_when_no_choices(self):
        user = _make_user(onboarding_choices=[])
        block = _build_onboarding_block(user)
        assert block == ""

    def test_includes_chosen_labels(self):
        user = _make_user(onboarding_choices=[
            {"chosen_label": "Spicy Beef Taco", "chosen_tags": ["spicy", "fried"]},
            {"chosen_label": "Chocolate Lava Cake", "chosen_tags": ["sweet", "rich"]},
        ])
        block = _build_onboarding_block(user)
        assert "Spicy Beef Taco" in block
        assert "Chocolate Lava Cake" in block
        assert "spicy" in block


class TestBuildCandidateBlock:
    def test_includes_all_item_ids(self):
        items = [_make_item("Wings"), _make_item("Salad")]
        block = _build_candidate_block(items)
        for item in items:
            assert str(item.id) in block

    def test_includes_taste_features(self):
        item = _make_item("Wings", spicy=0.9)
        block = _build_candidate_block([item])
        assert "spicy" in block


class TestBuildContextBlock:
    def test_includes_meal_intent_and_time(self):
        session = _make_session("full_meal")
        block = _build_context_block(session)
        assert "full_meal" in block
        assert "evening" in block

    def test_includes_budget(self):
        session = _make_session()
        block = _build_context_block(session)
        assert "50" in block


class TestRerankSingleItems:
    def test_raises_when_openai_key_missing(self, monkeypatch):
        monkeypatch.setattr(llm_svc.settings, "OPENAI_API_KEY", None)
        items = [_make_item("Wings")]
        user = _make_user()
        session = _make_session()
        with pytest.raises(LLMRecommendationError):
            rerank_single_items(items, user, session, [], {}, top_n=5)

    def test_raises_on_invalid_json_response(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text="not valid json",
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        items = [_make_item("Wings")]
        user = _make_user()
        session = _make_session()
        with pytest.raises(LLMRecommendationError, match="invalid JSON"):
            rerank_single_items(items, user, session, [], {}, top_n=5)

    def test_raises_on_empty_ranked_items(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text='{"ranked_items": []}',
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        items = [_make_item("Wings")]
        user = _make_user()
        session = _make_session()
        with pytest.raises(LLMRecommendationError, match="empty"):
            rerank_single_items(items, user, session, [], {}, top_n=5)

    def test_successful_rerank_returns_ordered_items(self, monkeypatch):
        item_a = _make_item("Spicy Wings", spicy=0.9)
        item_b = _make_item("Caesar Salad", spicy=0.1)
        items = [item_a, item_b]
        items_map = {str(i.id): i for i in items}

        mock_response = {
            "ranked_items": [
                {"item_id": str(item_a.id), "reason": "Matches spicy preference"},
                {"item_id": str(item_b.id), "reason": "Light option"},
            ]
        }
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text=json.dumps(mock_response),
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        user = _make_user(spicy_score=0.9)
        session = _make_session()
        result = rerank_single_items(items, user, session, [], items_map, top_n=5)

        assert len(result) == 2
        assert result[0]["item_id"] == str(item_a.id)
        assert result[1]["item_id"] == str(item_b.id)
        assert "spicy" in result[0]["reason"].lower()

    def test_missing_items_appended_at_end(self, monkeypatch):
        item_a = _make_item("Wings")
        item_b = _make_item("Salad")
        items = [item_a, item_b]

        mock_response = {
            "ranked_items": [
                {"item_id": str(item_a.id), "reason": "Good match"},
            ]
        }
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text=json.dumps(mock_response),
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        user = _make_user()
        session = _make_session()
        items_map = {str(i.id): i for i in items}
        result = rerank_single_items(items, user, session, [], items_map, top_n=5)

        assert len(result) == 2
        assert result[0]["item_id"] == str(item_a.id)
        assert result[1]["item_id"] == str(item_b.id)

    def test_respects_top_n_limit(self, monkeypatch):
        items = [_make_item(f"Item {i}") for i in range(10)]
        mock_ranked = [{"item_id": str(item.id), "reason": "ok"} for item in items]

        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text=json.dumps({"ranked_items": mock_ranked}),
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        user = _make_user()
        session = _make_session()
        items_map = {str(i.id): i for i in items}
        result = rerank_single_items(items, user, session, [], items_map, top_n=3)

        assert len(result) == 3

    def test_empty_candidates_returns_empty(self):
        user = _make_user()
        session = _make_session()
        result = rerank_single_items([], user, session, [], {}, top_n=5)
        assert result == []


class TestComposeFullMeal:
    def test_raises_on_invalid_json(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text="garbage",
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        items = [_make_item("App", "appetizer"), _make_item("Main"), _make_item("Cake", "dessert")]
        user = _make_user()
        session = _make_session("full_meal")
        with pytest.raises(LLMRecommendationError):
            compose_full_meal(items, user, session, [], {})

    def test_successful_composition(self, monkeypatch):
        app = _make_item("Soup", "appetizer")
        main = _make_item("Steak", "main")
        dessert = _make_item("Cake", "dessert")
        items = [app, main, dessert]
        items_map = {str(i.id): i for i in items}

        mock_response = {
            "compositions": [{
                "appetizer_id": str(app.id),
                "main_id": str(main.id),
                "dessert_id": str(dessert.id),
                "meal_reasoning": "Classic progression from light to rich",
            }]
        }
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text=json.dumps(mock_response),
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        user = _make_user()
        session = _make_session("full_meal")
        result = compose_full_meal(items, user, session, [], items_map)

        assert len(result) >= 1
        comp = result[0]
        assert comp["appetizer_id"] == str(app.id)
        assert comp["main_id"] == str(main.id)
        assert comp["dessert_id"] == str(dessert.id)

    def test_invalid_item_ids_rejected(self, monkeypatch):
        items = [_make_item("Soup", "appetizer")]

        mock_response = {
            "compositions": [{
                "appetizer_id": str(uuid4()),
                "main_id": str(uuid4()),
                "dessert_id": str(uuid4()),
                "meal_reasoning": "Made up IDs",
            }]
        }
        mock_client = MagicMock()
        mock_client.responses.create.return_value = MagicMock(
            output_text=json.dumps(mock_response),
            status="completed",
        )
        monkeypatch.setattr(llm_svc, "_openai_client", lambda: mock_client)

        user = _make_user()
        session = _make_session("full_meal")
        with pytest.raises(LLMRecommendationError, match="invalid item IDs"):
            compose_full_meal(items, user, session, [], {})

    def test_empty_candidates_returns_empty(self):
        user = _make_user()
        session = _make_session("full_meal")
        result = compose_full_meal([], user, session, [], {})
        assert result == []


class TestLLMRecommendationErrorIsRaised:
    def test_error_is_exception_subclass(self):
        assert issubclass(LLMRecommendationError, Exception)

    def test_error_message_preserved(self):
        err = LLMRecommendationError("test failure")
        assert str(err) == "test failure"
