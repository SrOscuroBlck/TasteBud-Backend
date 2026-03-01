from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from sqlmodel import Session, select
from sqlalchemy.orm.attributes import flag_modified

from models import User, MenuItem, Rating, Interaction, BayesianTasteProfile
from models.session import RecommendationFeedback, FeedbackType
from services.features.features import clamp01
from services.user.interaction_history_service import InteractionHistoryService
from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)


LEARNING_RATE_MAP = {
    "mild": 0.08,
    "medium": 0.15,
    "strong": 0.25
}


def temporal_weight(feedback_time: datetime, half_life_days: int = None) -> float:
    if not feedback_time:
        return 1.0
    
    if half_life_days is None:
        half_life_days = settings.FEEDBACK_HALF_LIFE_DAYS
    
    delta_seconds = (datetime.utcnow() - feedback_time).total_seconds()
    delta_days = delta_seconds / 86400.0
    
    weight = 0.5 ** (delta_days / half_life_days)
    
    return weight


class UnifiedFeedbackService:
    def __init__(self):
        self.interaction_history_service = InteractionHistoryService()
        self.use_bayesian_updates = True
    
    def record_session_feedback(
        self,
        db_session: Session,
        user: User,
        item: MenuItem,
        feedback_type: FeedbackType,
        session_id: UUID,
        comment: Optional[str] = None
    ) -> RecommendationFeedback:
        if not user:
            raise ValueError("user is required for feedback processing")
        
        if not item:
            raise ValueError("item is required for feedback processing")
        
        if not feedback_type:
            raise ValueError("feedback_type is required for feedback processing")
        
        intensity = self._get_feedback_intensity(feedback_type)
        
        feedback = RecommendationFeedback(
            session_id=session_id,
            item_id=item.id,
            feedback_type=feedback_type.value,
            comment=comment,
            timestamp=datetime.utcnow()
        )
        
        db_session.add(feedback)
        
        bayesian_profile = None
        if self.use_bayesian_updates:
            statement = select(BayesianTasteProfile).where(
                BayesianTasteProfile.user_id == user.id
            )
            bayesian_profile = db_session.exec(statement).first()
        
        if bayesian_profile:
            from services.learning.bayesian_profile_service import BayesianProfileService
            service = BayesianProfileService()
            service.update_from_feedback(
                db_session,
                bayesian_profile,
                item,
                feedback_type,
                feedback.timestamp
            )
            if feedback_type in [FeedbackType.DISLIKE, FeedbackType.SKIP] and item.ingredients:
                learning_rate = LEARNING_RATE_MAP.get(intensity, 0.05)
                self._track_disliked_ingredients(user, item, learning_rate * 2.0)
        else:
            self._update_user_profile(
                user=user,
                item=item,
                feedback_type=feedback_type,
                intensity=intensity
            )
        
        user.last_updated = datetime.utcnow()
        db_session.add(user)
        db_session.commit()
        db_session.refresh(feedback)
        
        try:
            self.interaction_history_service.update_interaction_outcome(
                db_session=db_session,
                user_id=user.id,
                item_id=item.id,
                was_dismissed=(feedback_type == FeedbackType.SKIP),
                was_disliked=(feedback_type == FeedbackType.DISLIKE),
                was_liked=(feedback_type == FeedbackType.LIKE),
                was_ordered=(feedback_type in [FeedbackType.SELECTED, FeedbackType.ACCEPTED])
            )
        except Exception as e:
            logger.warning(
                "Failed to update interaction history outcome",
                extra={"error": str(e), "item_id": str(item.id)}
            )
        
        logger.info(
            "Session feedback recorded and profile updated",
            extra={
                "user_id": str(user.id),
                "item_id": str(item.id),
                "feedback_type": feedback_type.value,
                "intensity": intensity,
                "session_id": str(session_id)
            }
        )
        
        return feedback
    
    def _get_feedback_intensity(self, feedback_type: FeedbackType) -> str:
        if feedback_type == FeedbackType.LIKE:
            return "mild"
        elif feedback_type == FeedbackType.SAVE_FOR_LATER:
            return "mild"
        elif feedback_type == FeedbackType.DISLIKE:
            return "strong"
        elif feedback_type == FeedbackType.SKIP:
            return "strong"
        elif feedback_type == FeedbackType.SELECTED:
            return "strong"
        return "medium"
    
    def _update_user_profile(
        self,
        user: User,
        item: MenuItem,
        feedback_type: FeedbackType,
        intensity: str
    ) -> None:
        learning_rate = LEARNING_RATE_MAP.get(intensity, 0.05)
        
        is_positive = feedback_type in [FeedbackType.LIKE, FeedbackType.SELECTED, FeedbackType.SAVE_FOR_LATER]
        is_negative = feedback_type in [FeedbackType.DISLIKE, FeedbackType.SKIP]
        
        if is_positive:
            for axis, value in item.features.items():
                if axis in user.taste_vector and value > 0.5:
                    delta = learning_rate * value
                    user.taste_vector[axis] = clamp01(user.taste_vector[axis] + delta)
                    user.taste_uncertainty[axis] = max(0.0, user.taste_uncertainty.get(axis, 0.5) - abs(delta))
            
            flag_modified(user, "taste_vector")
            flag_modified(user, "taste_uncertainty")
            
            for cuisine in item.cuisine:
                current = user.cuisine_affinity.get(cuisine, 0.5)
                user.cuisine_affinity[cuisine] = clamp01(current + learning_rate)
            
            flag_modified(user, "cuisine_affinity")
        
        elif is_negative:
            # AGGRESSIVE NEGATIVE LEARNING
            # When user dislikes/skips, they're VERY certain about not wanting this
            # Apply stronger penalties to taste dimensions and cuisine
            negative_multiplier = 2.0  # Double the learning rate for negative signals
            
            for axis, value in item.features.items():
                if axis in user.taste_vector and value > 0.5:
                    delta = learning_rate * value * negative_multiplier
                    user.taste_vector[axis] = clamp01(user.taste_vector[axis] - delta)
                    user.taste_uncertainty[axis] = max(0.0, user.taste_uncertainty.get(axis, 0.5) - abs(delta))
            
            flag_modified(user, "taste_vector")
            flag_modified(user, "taste_uncertainty")
            
            # Stronger cuisine penalty for explicit rejection
            for cuisine in item.cuisine:
                current = user.cuisine_affinity.get(cuisine, 0.5)
                user.cuisine_affinity[cuisine] = clamp01(current - learning_rate * negative_multiplier)
            
            flag_modified(user, "cuisine_affinity")
            
            # Ingredient-level learning: track disliked ingredients for cross-restaurant learning
            if item.ingredients:
                self._track_disliked_ingredients(user, item, learning_rate * negative_multiplier)
            
            logger.info(
                "Applied aggressive negative learning",
                extra={
                    "user_id": str(user.id),
                    "item_id": str(item.id),
                    "intensity": intensity,
                    "feedback_type": feedback_type.value,
                    "negative_multiplier": negative_multiplier,
                    "learning_rate": learning_rate
                }
            )

        
        logger.info(
            "User profile updated from feedback",
            extra={
                "user_id": str(user.id),
                "feedback_type": feedback_type.value,
                "intensity": intensity,
                "is_positive": is_positive,
                "is_negative": is_negative
            }
        )
    
    def _track_disliked_ingredients(
        self,
        user: User,
        item: MenuItem,
        penalty_strength: float
    ) -> None:
        if not hasattr(user, "ingredient_penalties") or user.ingredient_penalties is None:
            user.ingredient_penalties = {}
        
        # Track disliked ingredients for cross-restaurant learning
        # This helps prevent recommending items with similar ingredients
        for ingredient in item.ingredients[:10]:  # Top 10 ingredients only
            ingredient_lower = ingredient.lower().strip()
            if ingredient_lower:
                current_penalty = user.ingredient_penalties.get(ingredient_lower, 0.0)
                user.ingredient_penalties[ingredient_lower] = min(1.0, current_penalty + penalty_strength)
        
        flag_modified(user, "ingredient_penalties")
        
        logger.info(
            "Tracked disliked ingredients",
            extra={
                "user_id": str(user.id),
                "item_id": str(item.id),
                "ingredients_tracked": len(item.ingredients[:10]),
                "penalty_strength": penalty_strength
            }
        )
    
    def add_rating(
        self,
        db_session: Session,
        user: User,
        item_id: str,
        rating: int,
        liked: bool,
        reasons: List[str],
        comment: str = ""
    ) -> Rating:
        item = db_session.get(MenuItem, UUID(item_id))
        if not item:
            raise ValueError(f"MenuItem {item_id} not found")
        
        rating_record = Rating(
            user_id=user.id,
            item_id=item.id,
            rating=rating,
            liked=liked,
            reasons=",".join(reasons),
            comment=comment
        )
        db_session.add(rating_record)
        
        is_positive = liked or rating >= 4
        is_negative = rating <= 2
        intensity = "strong" if is_positive else "medium" if is_negative else "mild"
        learning_rate = LEARNING_RATE_MAP.get(intensity, 0.05)
        
        if is_positive:
            for axis, value in item.features.items():
                if axis in user.taste_vector and value > 0.5:
                    delta = learning_rate * value
                    user.taste_vector[axis] = clamp01(user.taste_vector[axis] + delta)
                    user.taste_uncertainty[axis] = max(0.0, user.taste_uncertainty.get(axis, 0.5) - abs(delta))
            
            flag_modified(user, "taste_vector")
            flag_modified(user, "taste_uncertainty")
            
            for cuisine in item.cuisine:
                current = user.cuisine_affinity.get(cuisine, 0.5)
                user.cuisine_affinity[cuisine] = clamp01(current + learning_rate)
            
            flag_modified(user, "cuisine_affinity")
        
        elif is_negative:
            for axis, value in item.features.items():
                if axis in user.taste_vector and value > 0.5:
                    delta = learning_rate * value
                    user.taste_vector[axis] = clamp01(user.taste_vector[axis] - delta)
                    user.taste_uncertainty[axis] = max(0.0, user.taste_uncertainty.get(axis, 0.5) - abs(delta))
            
            flag_modified(user, "taste_vector")
            flag_modified(user, "taste_uncertainty")
            
            for cuisine in item.cuisine:
                current = user.cuisine_affinity.get(cuisine, 0.5)
                user.cuisine_affinity[cuisine] = clamp01(current - learning_rate * 0.5)
            
            flag_modified(user, "cuisine_affinity")
        
        user.last_updated = datetime.utcnow()
        db_session.add(user)
        db_session.commit()
        db_session.refresh(rating_record)
        
        logger.info(
            "Direct rating recorded and profile updated",
            extra={
                "user_id": str(user.id),
                "item_id": str(item.id),
                "rating": rating,
                "liked": liked,
                "intensity": intensity
            }
        )
        
        return rating_record
    
    def add_interaction(
        self,
        db_session: Session,
        user: User,
        item_id: str,
        interaction_type: str
    ) -> Interaction:
        item = db_session.get(MenuItem, UUID(item_id))
        if not item:
            raise ValueError(f"MenuItem {item_id} not found")
        
        interaction = Interaction(
            user_id=user.id,
            item_id=item.id,
            type=interaction_type
        )
        db_session.add(interaction)
        db_session.commit()
        db_session.refresh(interaction)
        
        logger.info(
            "Interaction recorded",
            extra={
                "user_id": str(user.id),
                "item_id": str(item.id),
                "type": interaction_type
            }
        )
        
        return interaction
