from __future__ import annotations
from typing import Dict, List
from collections import Counter

from models import User, MenuItem, RecommendationSession, RecommendationFeedback
from models.user import TASTE_AXES
from services.features.features import cosine_similarity
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InSessionLearningService:
    def calculate_session_adjustments(
        self,
        user: User,
        session_feedback: List[RecommendationFeedback],
        session: RecommendationSession,
        items_map: Dict[str, MenuItem] | None = None,
    ) -> Dict[str, float]:
        if not session_feedback:
            return {}

        adjustments = {axis: 0.0 for axis in TASTE_AXES}
        if items_map is None:
            items_map = {}

        liked_items = [
            fb for fb in session_feedback
            if fb.feedback_type in ["like", "save_for_later"]
        ]
        disliked_items = [
            fb for fb in session_feedback
            if fb.feedback_type == "dislike"
        ]

        for feedback in liked_items:
            item = items_map.get(str(feedback.item_id))
            if not item:
                continue
            for axis, value in item.features.items():
                if axis in adjustments and value > 0.4:
                    adjustments[axis] += 0.15 * value

        for feedback in disliked_items:
            item = items_map.get(str(feedback.item_id))
            if not item:
                continue
            for axis, value in item.features.items():
                if axis in adjustments and value > 0.4:
                    adjustments[axis] -= 0.20 * value

        for axis in adjustments:
            adjustments[axis] = max(-0.5, min(0.5, adjustments[axis]))

        logger.info(
            "Calculating session adjustments",
            extra={
                "session_id": str(session.id),
                "liked": len(liked_items),
                "disliked": len(disliked_items),
                "adjustments": {k: round(v, 3) for k, v in adjustments.items() if v != 0.0},
            },
        )

        return adjustments
    
    def apply_immediate_learning(
        self,
        user_taste_vector: Dict[str, float],
        item_features: Dict[str, float],
        feedback_type: str,
        weight: float = 0.1
    ) -> Dict[str, float]:
        # Start with Phase 1 compliant axes
        adjusted = {axis: user_taste_vector.get(axis, 0.5) for axis in TASTE_AXES}
        
        if feedback_type == "dislike":
            for axis, value in item_features.items():
                if axis in TASTE_AXES and value > 0.6:
                    adjusted[axis] = max(0.0, adjusted[axis] - weight * value)
        
        elif feedback_type in ["like", "save_for_later"]:
            for axis, value in item_features.items():
                if axis in TASTE_AXES and value > 0.6:
                    adjusted[axis] = min(1.0, adjusted[axis] + weight * 0.5 * value)
        
        return adjusted
    
    def boost_similar_to_liked(
        self,
        candidates: List[MenuItem],
        liked_items: List[MenuItem],
        boost_factor: float = 0.2
    ) -> Dict[str, float]:
        if not liked_items:
            return {}
        
        boosts = {}
        
        for candidate in candidates:
            max_similarity = 0.0
            
            for liked in liked_items:
                similarity = cosine_similarity(candidate.features, liked.features)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity > 0.7:
                boosts[str(candidate.id)] = boost_factor * max_similarity
        
        return boosts
    
    def detect_cuisine_preferences(
        self,
        session_feedback: List[RecommendationFeedback],
        items_map: Dict[str, MenuItem]
    ) -> Dict[str, float]:
        liked_cuisines = []
        disliked_cuisines = []
        
        for feedback in session_feedback:
            item = items_map.get(str(feedback.item_id))
            if not item:
                continue
            
            if feedback.feedback_type in ["like", "save_for_later"]:
                liked_cuisines.extend(item.cuisine)
            elif feedback.feedback_type == "dislike":
                disliked_cuisines.extend(item.cuisine)
        
        cuisine_adjustments = {}
        
        if liked_cuisines:
            for cuisine, count in Counter(liked_cuisines).items():
                if count >= 2:
                    cuisine_adjustments[cuisine] = 0.15 * count
        
        if disliked_cuisines:
            for cuisine, count in Counter(disliked_cuisines).items():
                if count >= 2:
                    cuisine_adjustments[cuisine] = -0.15 * count
        
        return cuisine_adjustments
    
    def get_temporary_profile_adjustments(
        self,
        user: User,
        session_feedback: List[RecommendationFeedback],
        items_map: Dict[str, MenuItem]
    ) -> Dict[str, any]:
        # Use TASTE_AXES to ensure we only use Phase 1 compliant axis names
        taste_adjustments = {axis: 0.0 for axis in TASTE_AXES}
        cuisine_adjustments = self.detect_cuisine_preferences(session_feedback, items_map)
        
        disliked_items = [
            fb for fb in session_feedback
            if fb.feedback_type == "dislike"
        ]
        
        for feedback in disliked_items:
            item = items_map.get(str(feedback.item_id))
            if not item:
                continue

            for axis, value in item.features.items():
                if axis in taste_adjustments and value > 0.4:
                    taste_adjustments[axis] -= 0.20 * value

        liked_items = [
            fb for fb in session_feedback
            if fb.feedback_type in ["like", "save_for_later"]
        ]

        for feedback in liked_items:
            item = items_map.get(str(feedback.item_id))
            if not item:
                continue

            for axis, value in item.features.items():
                if axis in taste_adjustments and value > 0.4:
                    taste_adjustments[axis] += 0.15 * value

        for axis in taste_adjustments:
            taste_adjustments[axis] = max(-0.5, min(0.5, taste_adjustments[axis]))
        
        return {
            "taste_adjustments": taste_adjustments,
            "cuisine_adjustments": cuisine_adjustments
        }
