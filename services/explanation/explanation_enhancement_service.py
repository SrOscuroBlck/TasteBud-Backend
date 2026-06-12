from __future__ import annotations
from typing import Dict, List, Optional

from models import MenuItem, User, RecommendationSession, UserOrderHistory
from utils.logger import setup_logger

logger = setup_logger(__name__)

# All template fragments, fully localized. Every phrase the user can see must
# exist in every language — mixing languages in one explanation reads as a bug.
STRINGS: Dict[str, Dict[str, str]] = {
    "en": {
        "looks_great": "{name} looks great!",
        "still_learning": " We're still learning your taste, so let us know what you think!",
        "intent_appetizer": "Great starter choice!",
        "intent_main": "Perfect main dish",
        "intent_dessert": "Sweet way to end your meal!",
        "intent_beverage": "Refreshing choice!",
        "intent_snack": "Perfect light bite",
        "time_morning": "Perfect for breakfast!",
        "time_afternoon": "Ideal lunch option!",
        "time_evening": "Perfect for dinner!",
        "time_late_afternoon": "Great afternoon pick!",
        "time_night": "Nice late-night choice!",
        "axes_love": "This has those {axes} flavors you love!",
        "axes_match": "Features {axes} notes that match your profile.",
        "axes_joiner": " and ",
        "strong_match": "Strong match with your taste preferences!",
        "good_match": "Good match for your taste profile.",
        "similar_past": "Similar to dishes you've enjoyed before!",
        "reminds_favorites": "Reminds us of your favorites!",
        "great_value": "Great value",
        "fits_budget": "Fits your budget",
        "hearty": "hearty and satisfying",
        "light": "light and perfect",
        "mood_adventurous": "Perfect for trying something new!",
        "mood_comfort": "Comforting and familiar.",
        "mood_healthy": "Wholesome and nutritious.",
        "mood_indulgent": "A delicious treat!",
        "harmony_high": "perfectly balanced",
        "harmony_mid": "well-coordinated",
        "harmony_low": "diverse",
        "dining_experience": "A {quality} {minutes}-minute dining experience",
        "all_for": ", all for ${price}",
        "occasion_date_night": "Perfect for a romantic evening!",
        "occasion_celebration": "Great for celebrating!",
        "occasion_business": "Professional and refined.",
        "occasion_casual": "Relaxed and enjoyable.",
    },
    "es": {
        "looks_great": "¡{name} se ve genial!",
        "still_learning": " Aún estamos aprendiendo tus gustos, ¡cuéntanos qué te parece!",
        "intent_appetizer": "¡Excelente entrada!",
        "intent_main": "Plato fuerte perfecto",
        "intent_dessert": "¡Una forma dulce de terminar tu comida!",
        "intent_beverage": "¡Una opción refrescante!",
        "intent_snack": "El bocado ligero perfecto",
        "time_morning": "¡Perfecto para el desayuno!",
        "time_afternoon": "¡Ideal para el almuerzo!",
        "time_evening": "¡Perfecto para la cena!",
        "time_late_afternoon": "¡Gran elección para la tarde!",
        "time_night": "¡Buena opción para la noche!",
        "axes_love": "¡Tiene esos sabores {axes} que te encantan!",
        "axes_match": "Tiene notas {axes} que coinciden con tu perfil.",
        "axes_joiner": " y ",
        "strong_match": "¡Coincide muy bien con tus gustos!",
        "good_match": "Buena coincidencia con tu perfil de sabor.",
        "similar_past": "¡Parecido a platos que ya has disfrutado!",
        "reminds_favorites": "¡Nos recuerda a tus favoritos!",
        "great_value": "Excelente precio",
        "fits_budget": "Se ajusta a tu presupuesto",
        "hearty": "abundante y satisfactorio",
        "light": "ligero y perfecto",
        "mood_adventurous": "¡Perfecto para probar algo nuevo!",
        "mood_comfort": "Reconfortante y familiar.",
        "mood_healthy": "Saludable y nutritivo.",
        "mood_indulgent": "¡Un gusto delicioso!",
        "harmony_high": "perfectamente equilibrada",
        "harmony_mid": "bien coordinada",
        "harmony_low": "variada",
        "dining_experience": "Una experiencia gastronómica {quality} de {minutes} minutos",
        "all_for": ", todo por ${price}",
        "occasion_date_night": "¡Perfecto para una velada romántica!",
        "occasion_celebration": "¡Ideal para celebrar!",
        "occasion_business": "Profesional y refinado.",
        "occasion_casual": "Relajado y agradable.",
    },
}

# Taste axis display names (matches the frontend's flavors.* translations)
AXIS_NAMES: Dict[str, Dict[str, str]] = {
    "en": {},  # English uses the raw axis names
    "es": {
        "spicy": "picantes",
        "sweet": "dulces",
        "sour": "ácidas",
        "salty": "saladas",
        "bitter": "amargas",
        "umami": "umami",
        "fatty": "intensas",
        "rich": "intensas",
        "light": "ligeras",
        "savory": "saladas",
    },
}


def _resolve_lang(user: User) -> str:
    lang = getattr(user, "preferred_language", "en") or "en"
    return lang if lang in STRINGS else "en"


class ExplanationEnhancementService:
    def generate_personalized_explanation(
        self,
        item: MenuItem,
        user: User,
        context: RecommendationSession,
        ranking_factors: Dict[str, float],
        user_history: List[UserOrderHistory],
        confidence: float,
        restaurant_has_breakfast: bool = True
    ) -> str:
        lang = _resolve_lang(user)
        s = STRINGS[lang]
        parts = []

        time_context = self._get_time_context(
            context.time_of_day, context.meal_intent, s, restaurant_has_breakfast
        )
        if time_context:
            parts.append(time_context)

        taste_match = self._get_taste_match_text(item, user, ranking_factors, confidence, lang)
        if taste_match:
            parts.append(taste_match)

        past_reference = self._reference_past_items(item, user_history, s)
        if past_reference:
            parts.append(past_reference)

        context_fit = self._get_context_fit(item, context, s)
        if context_fit:
            parts.append(context_fit)

        if not parts:
            parts.append(s["looks_great"].format(name=item.name))

        explanation = " ".join(parts)

        explanation = self._add_personality(explanation, context.user_experience_level)

        if confidence < 0.7 and context.user_experience_level == "new":
            explanation += s["still_learning"]

        return explanation

    def _get_time_context(
        self, time_of_day: str, meal_intent: str, s: Dict[str, str], restaurant_has_breakfast: bool = True
    ) -> str:
        intent_phrases = {
            "appetizer": s["intent_appetizer"],
            "main": s["intent_main"],
            "dessert": s["intent_dessert"],
            "beverage": s["intent_beverage"],
            "snack": s["intent_snack"],
        }

        time_phrases = {
            "morning": s["time_morning"] if restaurant_has_breakfast else "",
            "afternoon": s["time_afternoon"],
            "evening": s["time_evening"],
            "late_afternoon": s["time_late_afternoon"],
            "night": s["time_night"],
        }

        if meal_intent in intent_phrases:
            return intent_phrases[meal_intent]
        elif time_of_day in time_phrases:
            return time_phrases[time_of_day]

        return ""

    def _get_taste_match_text(
        self,
        item: MenuItem,
        user: User,
        ranking_factors: Dict[str, float],
        confidence: float,
        lang: str
    ) -> str:
        s = STRINGS[lang]
        axis_names = AXIS_NAMES.get(lang, {})

        matched_axes = []
        for axis, user_pref in user.taste_vector.items():
            item_val = item.features.get(axis, 0.0)
            if user_pref > 0.6 and item_val > 0.6:
                matched_axes.append(axis)

        if len(matched_axes) >= 2:
            localized = [axis_names.get(a, a) for a in matched_axes[:2]]
            axes_text = s["axes_joiner"].join(localized)
            if confidence > 0.8:
                return s["axes_love"].format(axes=axes_text)
            else:
                return s["axes_match"].format(axes=axes_text)

        taste_similarity = ranking_factors.get("taste_similarity", 0.0)
        if taste_similarity > 0.8:
            return s["strong_match"]
        elif taste_similarity > 0.6:
            return s["good_match"]

        return ""

    def _reference_past_items(
        self,
        item: MenuItem,
        user_history: List[UserOrderHistory],
        s: Dict[str, str]
    ) -> str:
        if not user_history:
            return ""

        similar_past_items = []
        for order in user_history:
            if order.enjoyed and order.rating and order.rating >= 4:
                similar_past_items.append(order)

        if not similar_past_items:
            return ""

        if len(similar_past_items) >= 3:
            return s["similar_past"]
        elif len(similar_past_items) >= 1:
            return s["reminds_favorites"]

        return ""

    def _get_context_fit(self, item: MenuItem, context: RecommendationSession, s: Dict[str, str]) -> str:
        parts = []

        if context.budget and item.price:
            if item.price <= context.budget * 0.8:
                parts.append(s["great_value"])
            elif item.price <= context.budget:
                parts.append(s["fits_budget"])

        if context.hunger_level == "very_hungry":
            parts.append(s["hearty"])
        elif context.hunger_level == "light":
            parts.append(s["light"])

        if context.mood:
            mood_key = f"mood_{context.mood}"
            if mood_key in s:
                parts.append(s[mood_key])

        if parts:
            return " – " + ", ".join(parts) + "."

        return ""

    def _add_personality(self, explanation: str, user_experience_level: str) -> str:
        # Personality boosts were never localized; keep explanations as-is for
        # every experience level rather than injecting English-only phrases.
        return explanation

    def generate_multi_course_explanation(
        self,
        composition: any,
        user: User,
        context: RecommendationSession
    ) -> str:
        lang = _resolve_lang(user)
        s = STRINGS[lang]

        if composition.flavor_harmony_score > 0.8:
            harmony_quality = s["harmony_high"]
        elif composition.flavor_harmony_score > 0.6:
            harmony_quality = s["harmony_mid"]
        else:
            harmony_quality = s["harmony_low"]

        explanation = s["dining_experience"].format(
            quality=harmony_quality, minutes=composition.estimated_duration_minutes
        )

        if context.budget and composition.total_price <= context.budget:
            explanation += s["all_for"].format(price=f"{composition.total_price:.2f}")

        if context.occasion:
            occasion_key = f"occasion_{context.occasion}"
            if occasion_key in s:
                explanation += ". " + s[occasion_key]

        return explanation
