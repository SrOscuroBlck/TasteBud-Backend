from typing import Dict, List, Optional
from enum import Enum

from models import MenuItem, User
from services.features.gpt_helper import _client
from services.core.reranking_service import RankedItem
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ExplanationType(Enum):
    TASTE_MATCH = "taste_match"
    DIETARY_FIT = "dietary_fit"
    BUDGET_FRIENDLY = "budget_friendly"
    MOOD_ALIGNED = "mood_aligned"
    NEW_DISCOVERY = "new_discovery"
    POPULAR_CHOICE = "popular_choice"
    TIME_APPROPRIATE = "time_appropriate"


class ExplanationTemplates:
    TEMPLATES = {
        ExplanationType.TASTE_MATCH: [
            "{item_name} matches your love for {flavor_profile} flavors",
            "{item_name} aligns perfectly with your preference for {cuisine} cuisine",
            "{item_name} has the {taste_qualities} you enjoy",
        ],
        ExplanationType.DIETARY_FIT: [
            "{item_name} is {dietary_type} and perfect for your dietary needs",
            "{item_name} fits your {dietary_type} lifestyle perfectly",
            "Perfect match: {item_name} is {dietary_type}",
        ],
        ExplanationType.BUDGET_FRIENDLY: [
            "Great value at ${price} - {item_name} fits your budget perfectly",
            "{item_name} is an excellent choice within your ${budget} budget",
            "Budget-friendly: {item_name} at just ${price}",
        ],
        ExplanationType.MOOD_ALIGNED: [
            "{item_name} matches your {mood} mood with its {qualities}",
            "Perfect for feeling {mood}: {item_name}",
            "{item_name} brings the {mood} vibe you're looking for",
        ],
        ExplanationType.NEW_DISCOVERY: [
            "Try something new: {item_name} from {cuisine} cuisine",
            "Expand your palate with {item_name}",
            "Discover {item_name} - a unique {cuisine} dish",
        ],
        ExplanationType.POPULAR_CHOICE: [
            "{item_name} is highly rated by users with similar tastes",
            "Popular pick: {item_name} loved by many with your preferences",
            "{item_name} is a crowd favorite among similar taste profiles",
        ],
        ExplanationType.TIME_APPROPRIATE: [
            "{item_name} is perfect for {time_of_day}",
            "Great {time_of_day} choice: {item_name}",
            "{item_name} hits the spot for {time_of_day}",
        ],
    }
    
    @classmethod
    def get_template(cls, explanation_type: ExplanationType, index: int = 0) -> str:
        templates = cls.TEMPLATES.get(explanation_type, [])
        if not templates:
            return "{item_name} is recommended for you"
        return templates[index % len(templates)]


class ExplanationService:
    def __init__(self, use_llm_fallback: bool = True):
        self.use_llm_fallback = use_llm_fallback
        self.llm_call_count = 0
    
    def generate_explanations(
        self,
        ranked_items: List[RankedItem],
        user: User,
        context: Optional[Dict[str, any]] = None
    ) -> List[str]:
        explanations = []
        
        for idx, ranked_item in enumerate(ranked_items):
            explanation = self._generate_single_explanation(
                ranked_item, user, context, idx
            )
            explanations.append(explanation)
        
        return explanations
    
    def _generate_single_explanation(
        self,
        ranked_item: RankedItem,
        user: User,
        context: Optional[Dict[str, any]],
        position: int
    ) -> str:
        explanation_type = self._determine_explanation_type(
            ranked_item, user, context
        )
        
        try:
            explanation = self._render_template(
                explanation_type, ranked_item, user, context
            )
            return explanation
        except Exception as e:
            logger.warning(
                "Template rendering failed, trying LLM fallback",
                extra={"error": str(e), "item_id": str(ranked_item.item.id)}
            )
            
            if self.use_llm_fallback and self.llm_call_count < 10:
                return self._generate_with_llm(ranked_item, user, context)
            else:
                return f"{ranked_item.item.name} is recommended based on your preferences"
    
    def _determine_explanation_type(
        self,
        ranked_item: RankedItem,
        user: User,
        context: Optional[Dict[str, any]]
    ) -> ExplanationType:
        item = ranked_item.item
        factors = ranked_item.ranking_factors
        
        if context and context.get("time_of_day"):
            time_adj = factors.get("time_of_day_adjustment", 0.0)
            if time_adj > 0.1:
                return ExplanationType.TIME_APPROPRIATE
        
        dietary_match = any(
            tag.lower() in [d.lower() for d in user.dietary_rules]
            for tag in item.dietary_tags
        )
        if dietary_match:
            return ExplanationType.DIETARY_FIT
        
        if context and context.get("budget"):
            budget_adj = factors.get("budget_adjustment", 0.0)
            if budget_adj > 0:
                return ExplanationType.BUDGET_FRIENDLY
        
        if context and context.get("mood"):
            mood_adj = factors.get("mood_adjustment", 0.0)
            if mood_adj > 0.05:
                return ExplanationType.MOOD_ALIGNED
        
        cuisine_new = item.cuisine and not any(
            cuisine in user.cuisine_affinity
            for cuisine in item.cuisine
        )
        if cuisine_new:
            return ExplanationType.NEW_DISCOVERY
        
        popularity = factors.get("popularity", 0.0)
        if popularity > 0.7:
            return ExplanationType.POPULAR_CHOICE
        
        return ExplanationType.TASTE_MATCH
    
    def _render_template(
        self,
        explanation_type: ExplanationType,
        ranked_item: RankedItem,
        user: User,
        context: Optional[Dict[str, any]]
    ) -> str:
        item = ranked_item.item
        template = ExplanationTemplates.get_template(explanation_type)
        
        variables = {
            "item_name": item.name,
        }
        
        if explanation_type == ExplanationType.TASTE_MATCH:
            variables.update(self._get_taste_variables(item, user))
        
        elif explanation_type == ExplanationType.DIETARY_FIT:
            variables["dietary_type"] = self._get_dietary_match(item, user)
        
        elif explanation_type == ExplanationType.BUDGET_FRIENDLY:
            variables["price"] = f"{item.price:.2f}" if item.price else "N/A"
            if context and context.get("budget"):
                variables["budget"] = f"{context['budget']:.2f}"
        
        elif explanation_type == ExplanationType.MOOD_ALIGNED:
            variables["mood"] = context.get("mood", "great") if context else "great"
            variables["qualities"] = self._get_mood_qualities(item, context)
        
        elif explanation_type == ExplanationType.NEW_DISCOVERY:
            variables["cuisine"] = ", ".join(item.cuisine) if item.cuisine else "unique"
        
        elif explanation_type == ExplanationType.TIME_APPROPRIATE:
            variables["time_of_day"] = context.get("time_of_day", "now") if context else "now"
        
        try:
            return template.format(**variables)
        except KeyError as e:
            logger.warning(
                "Missing template variable",
                extra={"variable": str(e), "template": template}
            )
            return f"{item.name} is recommended based on your preferences"
    
    def _get_taste_variables(
        self,
        item: MenuItem,
        user: User
    ) -> Dict[str, str]:
        user_sorted = sorted(
            user.taste_vector.items(),
            key=lambda kv: kv[1],
            reverse=True
        )
        
        top_user_axes = [k for k, v in user_sorted if v > 0.6][:3]
        
        item_strong_axes = [
            axis for axis in top_user_axes
            if item.features.get(axis, 0.0) > 0.5
        ]
        
        if item_strong_axes:
            flavor_profile = ", ".join(item_strong_axes[:2])
        else:
            flavor_profile = "unique"
        
        cuisine_str = ", ".join(item.cuisine) if item.cuisine else "diverse"
        
        taste_qualities = flavor_profile
        
        return {
            "flavor_profile": flavor_profile,
            "cuisine": cuisine_str,
            "taste_qualities": taste_qualities,
        }
    
    def _get_dietary_match(self, item: MenuItem, user: User) -> str:
        user_diet_lower = set(d.lower() for d in user.dietary_rules)
        item_tags_lower = set(t.lower() for t in item.dietary_tags)
        
        matches = user_diet_lower.intersection(item_tags_lower)
        
        if matches:
            return ", ".join(matches)
        
        if item.dietary_tags:
            return item.dietary_tags[0]
        
        return "suitable"
    
    def _get_mood_qualities(
        self,
        item: MenuItem,
        context: Optional[Dict[str, any]]
    ) -> str:
        if not context or not context.get("mood"):
            return "delicious flavors"
        
        mood = context["mood"]
        
        if mood == "adventurous":
            if item.spice_level and item.spice_level >= 3:
                return "bold, spicy kick"
            return "exciting flavors"
        
        elif mood == "comfort":
            return "comforting, familiar taste"
        
        elif mood == "healthy":
            return "fresh, nutritious ingredients"
        
        return "great flavors"
    
    def _generate_with_llm(
        self,
        ranked_item: RankedItem,
        user: User,
        context: Optional[Dict[str, any]]
    ) -> str:
        self.llm_call_count += 1
        
        item = ranked_item.item
        
        user_sorted = sorted(
            user.taste_vector.items(),
            key=lambda kv: kv[1],
            reverse=True
        )
        top_preferences = [k for k, v in user_sorted[:3]]
        
        lang = getattr(user, "preferred_language", "en")
        lang_instruction = "Respond in Spanish." if lang == "es" else "Respond in English."

        prompt = f"""Generate a brief 1-sentence explanation (max 15 words) for why we recommended this dish to the user.

Dish: {item.name}
Cuisine: {', '.join(item.cuisine) if item.cuisine else 'N/A'}
Ingredients: {', '.join(item.ingredients[:5]) if item.ingredients else 'N/A'}

User preferences: {', '.join(top_preferences)}
User dietary: {', '.join(user.dietary_rules) if user.dietary_rules else 'None'}

Make it personal and specific. Start with the dish name. {lang_instruction}"""
        
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info(
                "LLM explanation generated",
                extra={"item_id": str(item.id), "llm_calls": self.llm_call_count}
            )
            return explanation
            
        except Exception as e:
            logger.error(
                "LLM explanation generation failed",
                extra={"error": str(e), "item_id": str(item.id)},
                exc_info=True
            )
            return f"{item.name} matches your taste preferences"
