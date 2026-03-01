from __future__ import annotations
from typing import Dict, List, Optional

from models import MenuItem, User, RecommendationSession, UserOrderHistory
from utils.logger import setup_logger

logger = setup_logger(__name__)


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
        parts = []
        
        time_context = self._get_time_context(
            context.time_of_day, context.meal_intent, restaurant_has_breakfast
        )
        if time_context:
            parts.append(time_context)
        
        taste_match = self._get_taste_match_text(item, user, ranking_factors, confidence)
        if taste_match:
            parts.append(taste_match)
        
        past_reference = self._reference_past_items(item, user_history)
        if past_reference:
            parts.append(past_reference)
        
        context_fit = self._get_context_fit(item, context)
        if context_fit:
            parts.append(context_fit)
        
        if not parts:
            parts.append(f"{item.name} looks great!")
        
        explanation = " ".join(parts)
        
        explanation = self._add_personality(explanation, context.user_experience_level)
        
        if confidence < 0.7 and context.user_experience_level == "new":
            explanation += " We're still learning your taste, so let us know what you think!"
        
        return explanation
    
    def _get_time_context(
        self, time_of_day: str, meal_intent: str, restaurant_has_breakfast: bool = True
    ) -> str:
        intent_phrases = {
            "appetizer": "Great starter choice!",
            "main": "Perfect main dish",
            "dessert": "Sweet way to end your meal!",
            "beverage": "Refreshing choice!",
            "snack": "Perfect light bite"
        }
        
        time_phrases = {
            "morning": "Perfect for breakfast!" if restaurant_has_breakfast else "",
            "afternoon": "Ideal lunch option!",
            "evening": "Perfect for dinner!",
            "late_afternoon": "Great afternoon pick!",
            "night": "Nice late-night choice!"
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
        confidence: float
    ) -> str:
        matched_axes = []
        for axis, user_pref in user.taste_vector.items():
            item_val = item.features.get(axis, 0.0)
            if user_pref > 0.6 and item_val > 0.6:
                matched_axes.append(axis)
        
        if len(matched_axes) >= 2:
            axes_text = " and ".join(matched_axes[:2])
            if confidence > 0.8:
                return f"This has those {axes_text} flavors you love!"
            else:
                return f"Features {axes_text} notes that match your profile."
        
        taste_similarity = ranking_factors.get("taste_similarity", 0.0)
        if taste_similarity > 0.8:
            return "Strong match with your taste preferences!"
        elif taste_similarity > 0.6:
            return "Good match for your taste profile."
        
        return ""
    
    def _reference_past_items(
        self,
        item: MenuItem,
        user_history: List[UserOrderHistory]
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
            return "Similar to dishes you've enjoyed before!"
        elif len(similar_past_items) >= 1:
            return "Reminds us of your favorites!"
        
        return ""
    
    def _get_context_fit(self, item: MenuItem, context: RecommendationSession) -> str:
        parts = []
        
        if context.budget and item.price:
            if item.price <= context.budget * 0.8:
                parts.append("Great value")
            elif item.price <= context.budget:
                parts.append("Fits your budget")
        
        if context.hunger_level == "very_hungry":
            parts.append("hearty and satisfying")
        elif context.hunger_level == "light":
            parts.append("light and perfect")
        
        if context.mood:
            mood_matches = {
                "adventurous": "Perfect for trying something new!",
                "comfort": "Comforting and familiar.",
                "healthy": "Wholesome and nutritious.",
                "indulgent": "A delicious treat!"
            }
            if context.mood in mood_matches:
                parts.append(mood_matches[context.mood])
        
        if parts:
            return " – " + ", ".join(parts) + "."
        
        return ""
    
    def _add_personality(self, explanation: str, user_experience_level: str) -> str:
        if user_experience_level == "new":
            return explanation
        
        if user_experience_level == "established":
            enthusiastic_starters = [
                "You're going to LOVE this!",
                "This is perfect for you!",
                "We found a great match!"
            ]
            
            if "love" in explanation.lower() or "perfect" in explanation.lower():
                return explanation
            
        return explanation
    
    def generate_multi_course_explanation(
        self,
        composition: any,
        user: User,
        context: RecommendationSession
    ) -> str:
        harmony_quality = ""
        if composition.flavor_harmony_score > 0.8:
            harmony_quality = "perfectly balanced"
        elif composition.flavor_harmony_score > 0.6:
            harmony_quality = "well-coordinated"
        else:
            harmony_quality = "diverse"
        
        explanation = f"A {harmony_quality} {composition.estimated_duration_minutes}-minute dining experience"
        
        if context.budget and composition.total_price <= context.budget:
            explanation += f", all for ${composition.total_price:.2f}"
        
        if context.occasion:
            occasion_text = {
                "date_night": "Perfect for a romantic evening!",
                "celebration": "Great for celebrating!",
                "business": "Professional and refined.",
                "casual": "Relaxed and enjoyable."
            }
            if context.occasion in occasion_text:
                explanation += ". " + occasion_text[context.occasion]
        
        return explanation
