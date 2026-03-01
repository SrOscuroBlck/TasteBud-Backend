from __future__ import annotations
import json
import re
from typing import Any, Dict
from uuid import UUID

from config.settings import settings
from models import MenuItem, User, RecommendationSession, RecommendationFeedback
from models.interaction_history import UserItemInteractionHistory
from models.session import UserOrderHistory
from utils.logger import setup_logger

logger = setup_logger(__name__)

MAX_CANDIDATES_FOR_LLM = 20
REASONING_EFFORT = "low"


class LLMRecommendationError(Exception):
    pass


def _openai_client():
    if not settings.OPENAI_API_KEY:
        raise LLMRecommendationError("OpenAI API key not configured")
    try:
        from openai import OpenAI
        return OpenAI(api_key=settings.OPENAI_API_KEY)
    except Exception as exc:
        raise LLMRecommendationError(f"Failed to create OpenAI client: {exc}") from exc


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _build_user_profile_block(user: User) -> str:
    top_taste = sorted(user.taste_vector.items(), key=lambda kv: kv[1], reverse=True)
    taste_summary = ", ".join(f"{axis}: {val:.2f}" for axis, val in top_taste)

    parts = [f"Taste profile (0=dislike, 1=love): {taste_summary}"]

    if user.cuisine_affinity:
        top_cuisines = sorted(user.cuisine_affinity.items(), key=lambda kv: kv[1], reverse=True)[:5]
        parts.append("Cuisine preferences: " + ", ".join(f"{c} ({v:.1f})" for c, v in top_cuisines))

    if user.liked_ingredients:
        parts.append(f"Likes: {', '.join(user.liked_ingredients[:10])}")
    if user.disliked_ingredients:
        parts.append(f"Dislikes: {', '.join(user.disliked_ingredients[:10])}")
    if user.allergies:
        parts.append(f"Allergies: {', '.join(user.allergies)}")
    if user.dietary_rules:
        parts.append(f"Diet: {', '.join(user.dietary_rules)}")

    return "\n".join(parts)


def _build_onboarding_block(user: User) -> str:
    choices = user.onboarding_choices or []
    if not choices:
        return ""
    lines = ["During onboarding the user made these food choices:"]
    for c in choices:
        label = c.get("chosen_label", "")
        tags = ", ".join(c.get("chosen_tags", []))
        if label:
            lines.append(f"- Chose \"{label}\" (tags: {tags})")
    return "\n".join(lines)


def _build_session_feedback_block(
    feedback_list: list[RecommendationFeedback],
    items_map: dict[str, MenuItem],
) -> str:
    if not feedback_list:
        return ""
    lines = ["In-session feedback so far:"]
    for fb in feedback_list:
        item = items_map.get(str(fb.item_id))
        name = item.name if item else str(fb.item_id)
        lines.append(f"- {fb.feedback_type.upper()}: \"{name}\"")
    return "\n".join(lines)


def _build_dislike_history_block(
    interaction_history: Dict[UUID, UserItemInteractionHistory],
    items_map: dict[str, MenuItem],
) -> str:
    if not interaction_history:
        return ""
    disliked_names: list[str] = []
    for item_id, history in interaction_history.items():
        if history.was_disliked:
            item = items_map.get(str(item_id))
            name = item.name if item else str(item_id)
            disliked_names.append(name)
    if not disliked_names:
        return ""
    lines = ["Items previously rejected by this user (from past visits):"]
    for name in disliked_names:
        lines.append(f"- REJECTED: \"{name}\"")
    lines.append(
        "Strongly deprioritize these items and items with similar flavor profiles or ingredients. "
        "Place them near the bottom of the ranking. However, do NOT completely exclude them — "
        "taste evolves over time, and they may resurface if nothing better fits."
    )
    return "\n".join(lines)


def _build_shown_items_block(
    items_shown: list[str],
    items_map: dict[str, MenuItem],
) -> str:
    if not items_shown:
        return ""
    shown_lines: list[str] = []
    for item_id_str in items_shown:
        item = items_map.get(item_id_str)
        if item:
            shown_lines.append(f'- ALREADY SHOWN: "{item.name}" (ID:{item.id})')
    if not shown_lines:
        return ""
    lines = [
        "These items were already shown to the user earlier in this session. "
        "Strongly prioritize NEW items they haven't seen yet:"
    ]
    lines.extend(shown_lines)
    return "\n".join(lines)


def _build_candidate_block(candidates: list[MenuItem]) -> str:
    lines: list[str] = []
    for item in candidates:
        price_str = f"${item.price:,.0f}" if item.price else "N/A"
        ingredients_str = ", ".join(item.ingredients[:8]) if item.ingredients else "unknown"
        cuisine_str = ", ".join(item.cuisine) if item.cuisine else "unknown"
        course_str = item.course or "unknown"

        taste_top = ""
        if item.features:
            top_feats = sorted(item.features.items(), key=lambda kv: kv[1], reverse=True)[:4]
            taste_top = " | taste: " + ", ".join(f"{k}={v:.1f}" for k, v in top_feats)

        lines.append(
            f"ID:{item.id} | \"{item.name}\" | {course_str} | {cuisine_str} | "
            f"{price_str} | ingredients: {ingredients_str}{taste_top}"
        )
    return "\n".join(lines)


def _build_order_history_block(
    order_history: list[UserOrderHistory],
    items_map: dict[str, MenuItem],
) -> str:
    if not order_history:
        return ""
    ordered_names: list[str] = []
    for order in order_history:
        item = items_map.get(str(order.item_id))
        name = item.name if item else str(order.item_id)
        enjoyed_tag = "(enjoyed)" if order.enjoyed else ""
        ordered_names.append(f'- PREVIOUSLY ORDERED: "{name}" {enjoyed_tag}'.strip())
    if not ordered_names:
        return ""
    lines = [
        "The user has ordered these dishes before at this restaurant. "
        "They are using the app to DISCOVER something new — do NOT recommend these "
        "unless there are very few candidates left:"
    ]
    lines.extend(ordered_names)
    return "\n".join(lines)


def _build_context_block(rec_session: RecommendationSession) -> str:
    parts = [
        f"Meal intent: {rec_session.meal_intent}",
        f"Time: {rec_session.time_of_day} (hour {rec_session.detected_hour})",
        f"Hunger: {rec_session.hunger_level}",
    ]
    if rec_session.budget:
        parts.append(f"Budget: ${rec_session.budget:,.0f}")
    if rec_session.mood:
        parts.append(f"Mood: {rec_session.mood}")
    if rec_session.occasion:
        parts.append(f"Occasion: {rec_session.occasion}")
    if rec_session.party_size > 1:
        parts.append(f"Party size: {rec_session.party_size}")
    return "\n".join(parts)


SINGLE_ITEM_SYSTEM_PROMPT = (
    "You are a food recommendation engine. Given the user's taste profile, their history, "
    "session context, and a list of candidate menu items, rank the candidates from BEST to WORST match "
    "for this specific user.\n\n"
    "Ranking criteria (in order of importance):\n"
    "1. NOVELTY & DISCOVERY — users come here to discover NEW dishes they haven't tried. "
    "Strongly prioritize items NOT yet shown this session. Previously shown items should rank lower.\n"
    "2. Taste profile alignment — items whose flavor characteristics match the user's taste_vector scores\n"
    "3. Onboarding choices — the foods they chose reveal concrete preferences\n"
    "4. Cross-session rejection history — items the user previously rejected should be ranked MUCH lower, "
    "and items with similar flavor profiles or ingredients to rejected items should also be deprioritized. "
    "But never completely exclude them — taste changes over time.\n"
    "5. In-session feedback — respect likes/dislikes expressed RIGHT NOW\n"
    "6. Context (time of day, mood, occasion, hunger level)\n"
    "7. Variety — don't put 3 similar items in a row\n\n"
    "Return ONLY a JSON object (no markdown, no explanation outside JSON):\n"
    "{\n"
    '  "ranked_items": [\n'
    '    {"item_id": "<uuid>", "reason": "<1 sentence — warm, conversational, explaining WHY this dish fits them personally. No numbers or scores.>"},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "Return ALL candidate item_ids, ordered best-first. "
    "The 'reason' must feel like a friend recommending food — warm, specific to THIS user, no technical jargon."
)

FULL_MEAL_SYSTEM_PROMPT = (
    "You are a food recommendation engine specializing in composing complete meals. "
    "Given the user's taste profile, their history, session context, and candidate menu items, "
    "compose {num_compositions} cohesive meal(s) of appetizer + main + dessert.\n\n"
    "Composition criteria:\n"
    "1. NOVELTY & DISCOVERY — users come here to discover NEW dishes. Strongly prefer items NOT already shown this session.\n"
    "2. Each meal should tell a flavor story — appetizer prepares the palate, main delivers, dessert complements\n"
    "3. Match the user's taste profile — a spicy-lover gets a spicy-forward meal, not a bland one\n"
    "4. Respect onboarding choices and in-session feedback\n"
    "5. Strongly avoid using items the user previously rejected in past visits, and avoid items "
    "with similar flavor profiles to rejected ones. Only include them if no better alternatives exist.\n"
    "6. Consider flavor progression (light→rich, mild→bold, etc.)\n"
    "7. Stay within budget if specified (sum of 3 items)\n"
    "8. Each composition should be DIFFERENT from the others\n\n"
    "Return ONLY a JSON object (no markdown, no explanation outside JSON):\n"
    "{\n"
    '  "compositions": [\n'
    "    {\n"
    '      "appetizer_id": "<uuid>",\n'
    '      "main_id": "<uuid>",\n'
    '      "dessert_id": "<uuid>",\n'
    '      "meal_reasoning": "<2 sentences — warm, conversational, explaining WHY these 3 work together for THIS user. No numbers or scores.>"\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Use ONLY item IDs from the candidate list. Each item may appear in at most one composition."
)


def rerank_single_items(
    candidates: list[MenuItem],
    user: User,
    rec_session: RecommendationSession,
    feedback_list: list[RecommendationFeedback],
    items_map: dict[str, MenuItem],
    top_n: int = 10,
    interaction_history: Dict[UUID, UserItemInteractionHistory] | None = None,
    items_shown: list[str] | None = None,
    order_history: list[UserOrderHistory] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    trimmed = candidates[:MAX_CANDIDATES_FOR_LLM]

    user_block = _build_user_profile_block(user)
    onboarding_block = _build_onboarding_block(user)
    feedback_block = _build_session_feedback_block(feedback_list, items_map)
    dislike_history_block = _build_dislike_history_block(interaction_history or {}, items_map)
    shown_items_block = _build_shown_items_block(items_shown or [], items_map)
    order_history_block = _build_order_history_block(order_history or [], items_map)
    context_block = _build_context_block(rec_session)
    candidate_block = _build_candidate_block(trimmed)

    user_prompt_parts = [
        "## User Profile",
        user_block,
    ]
    if onboarding_block:
        user_prompt_parts += ["", "## Onboarding History", onboarding_block]
    if order_history_block:
        user_prompt_parts += ["", "## Previous Orders at This Restaurant", order_history_block]
    if dislike_history_block:
        user_prompt_parts += ["", "## Past Rejection History", dislike_history_block]
    if shown_items_block:
        user_prompt_parts += ["", "## Items Already Shown This Session", shown_items_block]
    if feedback_block:
        user_prompt_parts += ["", "## Session Feedback", feedback_block]
    user_prompt_parts += [
        "",
        "## Context",
        context_block,
        "",
        "## Candidates",
        candidate_block,
        "",
        f"Rank all {len(trimmed)} items for this user, best first. Return valid JSON only.",
    ]
    user_prompt = "\n".join(user_prompt_parts)

    client = _openai_client()

    logger.info(
        "LLM reranking request",
        extra={
            "user_id": str(user.id),
            "session_id": str(rec_session.id),
            "candidate_count": len(trimmed),
            "items_shown_count": len(items_shown or []),
            "mode": "single_item",
        },
    )

    try:
        response = client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=SINGLE_ITEM_SYSTEM_PROMPT,
            input=user_prompt,
            reasoning={"effort": REASONING_EFFORT},
            text={"format": {"type": "json_object"}},
        )
    except Exception as exc:
        raise LLMRecommendationError(f"OpenAI API call failed: {exc}") from exc

    if response.status == "incomplete":
        logger.warning(
            "LLM response incomplete",
            extra={"session_id": str(rec_session.id), "status": response.status},
        )

    raw = response.output_text or ""
    cleaned = _strip_markdown_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON", extra={"raw_response": raw[:500]})
        raise LLMRecommendationError("LLM returned invalid JSON") from exc

    ranked_items = data.get("ranked_items", [])
    if not ranked_items:
        raise LLMRecommendationError("LLM returned empty ranked_items")

    candidate_ids = {str(item.id) for item in trimmed}
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for entry in ranked_items:
        item_id = str(entry.get("item_id", ""))
        if item_id in candidate_ids and item_id not in seen_ids:
            validated.append({
                "item_id": item_id,
                "reason": entry.get("reason", ""),
            })
            seen_ids.add(item_id)

    missing = candidate_ids - seen_ids
    for mid in missing:
        validated.append({"item_id": mid, "reason": ""})

    logger.info(
        "LLM reranking completed",
        extra={
            "user_id": str(user.id),
            "session_id": str(rec_session.id),
            "returned_count": len(validated),
            "llm_returned": len(ranked_items),
            "missing_appended": len(missing),
        },
    )

    return validated[:top_n]


def compose_full_meal(
    candidates: list[MenuItem],
    user: User,
    rec_session: RecommendationSession,
    feedback_list: list[RecommendationFeedback],
    items_map: dict[str, MenuItem],
    num_compositions: int = 3,
    interaction_history: Dict[UUID, UserItemInteractionHistory] | None = None,
    items_shown: list[str] | None = None,
    order_history: list[UserOrderHistory] | None = None,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    trimmed = candidates[:MAX_CANDIDATES_FOR_LLM]

    user_block = _build_user_profile_block(user)
    onboarding_block = _build_onboarding_block(user)
    feedback_block = _build_session_feedback_block(feedback_list, items_map)
    dislike_history_block = _build_dislike_history_block(interaction_history or {}, items_map)
    shown_items_block = _build_shown_items_block(items_shown or [], items_map)
    order_history_block = _build_order_history_block(order_history or [], items_map)
    context_block = _build_context_block(rec_session)
    candidate_block = _build_candidate_block(trimmed)

    user_prompt_parts = [
        "## User Profile",
        user_block,
    ]
    if onboarding_block:
        user_prompt_parts += ["", "## Onboarding History", onboarding_block]
    if order_history_block:
        user_prompt_parts += ["", "## Previous Orders at This Restaurant", order_history_block]
    if dislike_history_block:
        user_prompt_parts += ["", "## Past Rejection History", dislike_history_block]
    if shown_items_block:
        user_prompt_parts += ["", "## Items Already Shown This Session", shown_items_block]
    if feedback_block:
        user_prompt_parts += ["", "## Session Feedback", feedback_block]
    user_prompt_parts += [
        "",
        "## Context",
        context_block,
        "",
        "## Candidates",
        candidate_block,
        "",
        f"Compose {num_compositions} complete meal(s) for this user. Each meal = 1 appetizer + 1 main + 1 dessert. Return valid JSON only.",
    ]
    user_prompt = "\n".join(user_prompt_parts)

    system_prompt = FULL_MEAL_SYSTEM_PROMPT.replace("{num_compositions}", str(num_compositions))
    client = _openai_client()

    logger.info(
        "LLM meal composition request",
        extra={
            "user_id": str(user.id),
            "session_id": str(rec_session.id),
            "candidate_count": len(trimmed),
            "num_compositions": num_compositions,
            "items_shown_count": len(items_shown or []),
            "mode": "full_meal",
        },
    )

    try:
        response = client.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            reasoning={"effort": REASONING_EFFORT},
            text={"format": {"type": "json_object"}},
        )
    except Exception as exc:
        raise LLMRecommendationError(f"OpenAI API call failed: {exc}") from exc

    if response.status == "incomplete":
        logger.warning(
            "LLM meal composition incomplete",
            extra={"session_id": str(rec_session.id), "status": response.status},
        )

    raw = response.output_text or ""
    cleaned = _strip_markdown_fences(raw)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("LLM returned invalid JSON for meal composition", extra={"raw_response": raw[:500]})
        raise LLMRecommendationError("LLM returned invalid JSON for meal composition") from exc

    compositions_raw = data.get("compositions", [])
    if not compositions_raw:
        raise LLMRecommendationError("LLM returned empty compositions")

    candidate_ids = {str(item.id) for item in trimmed}
    validated: list[dict[str, Any]] = []

    for comp in compositions_raw:
        app_id = str(comp.get("appetizer_id", ""))
        main_id = str(comp.get("main_id", ""))
        dessert_id = str(comp.get("dessert_id", ""))
        reasoning = comp.get("meal_reasoning", "")

        if app_id in candidate_ids and main_id in candidate_ids and dessert_id in candidate_ids:
            validated.append({
                "appetizer_id": app_id,
                "main_id": main_id,
                "dessert_id": dessert_id,
                "meal_reasoning": reasoning,
            })

    if not validated:
        raise LLMRecommendationError("LLM meal compositions contained invalid item IDs")

    logger.info(
        "LLM meal composition completed",
        extra={
            "user_id": str(user.id),
            "session_id": str(rec_session.id),
            "compositions_requested": num_compositions,
            "compositions_valid": len(validated),
        },
    )

    return validated
