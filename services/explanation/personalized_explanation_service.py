from typing import Dict, List, Optional
from datetime import datetime, timedelta
from uuid import UUID
from sqlmodel import Session, select
from sqlalchemy import desc

from models import MenuItem, User, Rating, Interaction
from models.interaction_history import UserItemInteractionHistory
from services.features.gpt_helper import _client
from services.core.reranking_service import RankedItem
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class UserHistory:
    def __init__(
        self,
        recent_likes: List[MenuItem],
        recent_dislikes: List[MenuItem],
        recent_orders: List[MenuItem],
        frequently_ordered_cuisines: List[str],
        favorite_taste_axes: List[str]
    ):
        self.recent_likes = recent_likes
        self.recent_dislikes = recent_dislikes
        self.recent_orders = recent_orders
        self.frequently_ordered_cuisines = frequently_ordered_cuisines
        self.favorite_taste_axes = favorite_taste_axes


class PersonalizedExplanationService:
    def __init__(self):
        self.cache_ttl_hours = 24
        self.max_history_items = 5
    
    def generate_explanations(
        self,
        session: Session,
        ranked_items: List[RankedItem],
        user: User,
        context: Optional[Dict[str, any]] = None
    ) -> List[str]:
        if not user:
            raise ValueError("user is required for explanation generation")
        
        user_history = self._fetch_user_history(session, user)
        
        explanations = []
        for idx, ranked_item in enumerate(ranked_items):
            explanation = self._generate_single_explanation(
                ranked_item, user, user_history, context, idx
            )
            explanations.append(explanation)
        
        return explanations
    
    def _fetch_user_history(self, session: Session, user: User) -> UserHistory:
        recent_cutoff = datetime.utcnow() - timedelta(days=30)
        
        recent_likes_stmt = (
            select(Rating)
            .where(Rating.user_id == user.id)
            .where(Rating.liked == True)
            .where(Rating.timestamp >= recent_cutoff)
            .order_by(desc(Rating.timestamp))
            .limit(self.max_history_items)
        )
        recent_like_ratings = session.exec(recent_likes_stmt).all()
        
        recent_likes = []
        for rating in recent_like_ratings:
            item_stmt = select(MenuItem).where(MenuItem.id == rating.item_id)
            item = session.exec(item_stmt).first()
            if item:
                recent_likes.append(item)
        
        recent_dislikes_stmt = (
            select(Rating)
            .where(Rating.user_id == user.id)
            .where(Rating.liked == False)
            .where(Rating.timestamp >= recent_cutoff)
            .order_by(desc(Rating.timestamp))
            .limit(self.max_history_items)
        )
        recent_dislike_ratings = session.exec(recent_dislikes_stmt).all()
        
        recent_dislikes = []
        for rating in recent_dislike_ratings:
            item_stmt = select(MenuItem).where(MenuItem.id == rating.item_id)
            item = session.exec(item_stmt).first()
            if item:
                recent_dislikes.append(item)
        
        interaction_stmt = (
            select(UserItemInteractionHistory)
            .where(UserItemInteractionHistory.user_id == user.id)
            .where(UserItemInteractionHistory.was_ordered == True)
            .order_by(desc(UserItemInteractionHistory.last_shown_at))
            .limit(self.max_history_items)
        )
        ordered_interactions = session.exec(interaction_stmt).all()
        
        recent_orders = []
        for interaction in ordered_interactions:
            item_stmt = select(MenuItem).where(MenuItem.id == interaction.item_id)
            item = session.exec(item_stmt).first()
            if item:
                recent_orders.append(item)
        
        cuisine_counts: Dict[str, int] = {}
        for item in recent_orders:
            if item.cuisine:
                for cuisine in item.cuisine:
                    cuisine_counts[cuisine] = cuisine_counts.get(cuisine, 0) + 1
        
        frequently_ordered_cuisines = sorted(
            cuisine_counts.items(),
            key=lambda kv: kv[1],
            reverse=True
        )[:3]
        frequently_ordered_cuisines = [cuisine for cuisine, _ in frequently_ordered_cuisines]
        
        favorite_taste_axes = []
        if user.taste_vector:
            sorted_axes = sorted(
                user.taste_vector.items(),
                key=lambda kv: kv[1],
                reverse=True
            )
            favorite_taste_axes = [axis for axis, value in sorted_axes if value > 0.6][:3]
        
        return UserHistory(
            recent_likes=recent_likes,
            recent_dislikes=recent_dislikes,
            recent_orders=recent_orders,
            frequently_ordered_cuisines=frequently_ordered_cuisines,
            favorite_taste_axes=favorite_taste_axes
        )
    
    def _generate_single_explanation(
        self,
        ranked_item: RankedItem,
        user: User,
        user_history: UserHistory,
        context: Optional[Dict[str, any]],
        position: int
    ) -> str:
        item = ranked_item.item
        ranking_factors = ranked_item.ranking_factors
        
        lang = getattr(user, "preferred_language", "en")
        system_prompt = self._build_system_prompt(language=lang)
        user_prompt = self._build_user_prompt(
            item, user, user_history, ranking_factors, context
        )
        
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=80,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            
            logger.info(
                "Personalized LLM explanation generated",
                extra={
                    "item_id": str(item.id),
                    "user_id": str(user.id),
                    "position": position
                }
            )
            
            return explanation
            
        except Exception as e:
            logger.error(
                "LLM explanation generation failed",
                extra={
                    "error": str(e),
                    "item_id": str(item.id),
                    "user_id": str(user.id)
                },
                exc_info=True
            )
            
            return self._generate_fallback_explanation(item, user, user_history)
    
    def _build_system_prompt(self, language: str = "en") -> str:
        if language == "es":
            return """Eres un asistente de recomendación gastronómica que crea explicaciones personalizadas y concisas para recomendaciones de platos del menú.

Tus explicaciones deben:
- Ser de 1-2 oraciones máximo (15-25 palabras en total)
- Hacer referencia a las preferencias e historial del usuario cuando sea relevante
- Ser conversacionales y naturales
- Comenzar con el nombre del plato o una referencia directa a él
- Explicar POR QUÉ este plato coincide con el gusto del usuario

Ejemplos:
- "El Spicy Tuna Roll combina con tu amor por los sabores intensos y umami, similar al Pad Thai que pediste la semana pasada."
- "Esta cremosa Carbonara se alinea con tu preferencia por platos italianos ricos y sustanciosos."
- "La Pizza Margherita es una opción clásica para tu paladar equilibrado, con las notas de albahaca fresca que disfrutas."

IMPORTANTE: Responde siempre en español.
"""
        return """You are a food recommendation assistant that creates personalized, concise explanations for menu item recommendations.

Your explanations should:
- Be 1-2 sentences maximum (15-25 words total)
- Reference specific user preferences and history when relevant
- Be conversational and natural
- Start with the dish name or a direct reference to it
- Explain WHY this dish matches the user's taste

Examples:
- "Spicy Tuna Roll matches your love for bold, umami flavors - similar to the Pad Thai you ordered last week."
- "This creamy Carbonara aligns with your preference for rich, fatty Italian dishes."
- "Margherita Pizza is a classic choice for your balanced palate, with the fresh basil notes you enjoy."
"""
    
    def _build_user_prompt(
        self,
        item: MenuItem,
        user: User,
        user_history: UserHistory,
        ranking_factors: Dict[str, float],
        context: Optional[Dict[str, any]]
    ) -> str:
        prompt_parts = [
            f"Dish: {item.name}",
            f"Cuisine: {', '.join(item.cuisine) if item.cuisine else 'N/A'}",
            f"Description: {item.description if item.description else 'N/A'}"
        ]
        
        if item.ingredients:
            ingredients_str = ', '.join(item.ingredients[:5])
            prompt_parts.append(f"Key ingredients: {ingredients_str}")
        
        if user_history.favorite_taste_axes:
            prompt_parts.append(
                f"User's favorite taste profiles: {', '.join(user_history.favorite_taste_axes)}"
            )
        
        if user_history.recent_likes:
            liked_names = [i.name for i in user_history.recent_likes[:3]]
            prompt_parts.append(
                f"User recently liked: {', '.join(liked_names)}"
            )
        
        if user_history.recent_orders:
            ordered_names = [i.name for i in user_history.recent_orders[:3]]
            prompt_parts.append(
                f"User recently ordered: {', '.join(ordered_names)}"
            )
        
        if user_history.frequently_ordered_cuisines:
            prompt_parts.append(
                f"User frequently orders: {', '.join(user_history.frequently_ordered_cuisines[:2])} cuisine"
            )
        
        if user.dietary_rules:
            prompt_parts.append(
                f"User's dietary preferences: {', '.join(user.dietary_rules)}"
            )
        
        if context:
            if context.get("mood"):
                prompt_parts.append(f"Current mood: {context['mood']}")
            if context.get("time_of_day"):
                prompt_parts.append(f"Time of day: {context['time_of_day']}")
        
        key_factors = []
        taste_sim = ranking_factors.get("taste_similarity", 0.0)
        if taste_sim > 0.6:
            key_factors.append("strong taste match")
        
        exploration_bonus = ranking_factors.get("exploration_bonus", 0.0)
        if exploration_bonus > 0.1:
            key_factors.append("new exploration")
        
        if key_factors:
            prompt_parts.append(f"Why recommended: {', '.join(key_factors)}")
        
        prompt = "\n".join(prompt_parts)
        lang = getattr(user, "preferred_language", "en")
        if lang == "es":
            prompt += "\n\nGenera una explicación personalizada de 1-2 oraciones (15-25 palabras) de por qué se recomienda este plato. Responde en español."
        else:
            prompt += "\n\nGenerate a personalized 1-2 sentence explanation (15-25 words) for why this dish is recommended."

        return prompt
    
    def _generate_fallback_explanation(
        self,
        item: MenuItem,
        user: User,
        user_history: UserHistory
    ) -> str:
        lang = getattr(user, "preferred_language", "en")
        if lang == "es":
            if user_history.favorite_taste_axes:
                axes_str = " y ".join(user_history.favorite_taste_axes[:2])
                return f"{item.name} combina con tu preferencia por sabores {axes_str}."
            if user_history.frequently_ordered_cuisines:
                cuisine = user_history.frequently_ordered_cuisines[0]
                return f"{item.name} se recomienda por tu gusto por la cocina {cuisine}."
            if item.cuisine:
                cuisine_str = ", ".join(item.cuisine[:2])
                return f"{item.name} ofrece excelentes sabores de cocina {cuisine_str} que podrías disfrutar."
            return f"{item.name} se recomienda según tu perfil de gustos."

        if user_history.favorite_taste_axes:
            axes_str = " and ".join(user_history.favorite_taste_axes[:2])
            return f"{item.name} matches your preference for {axes_str} flavors."

        if user_history.frequently_ordered_cuisines:
            cuisine = user_history.frequently_ordered_cuisines[0]
            return f"{item.name} is recommended based on your love for {cuisine} cuisine."

        if item.cuisine:
            cuisine_str = ", ".join(item.cuisine[:2])
            return f"{item.name} offers great {cuisine_str} flavors you might enjoy."

        return f"{item.name} is recommended based on your taste profile."
