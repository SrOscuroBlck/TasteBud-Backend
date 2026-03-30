from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime
from uuid import UUID
from sqlmodel import Session, select
from sqlalchemy.orm.attributes import flag_modified

from models import (
    User, Restaurant, RecommendationSession, Rating, Interaction,
    UserOrderHistory
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RecommendationSessionService:
    def start_session(
        self,
        db_session: Session,
        user_id: UUID,
        restaurant_id: UUID,
        meal_intent: str,
        budget: Optional[float] = None,
        time_constraint_minutes: Optional[int] = None,
        hunger_level: str = "moderate",
        mood: Optional[str] = None,
        occasion: Optional[str] = None,
        dietary_notes: Optional[str] = None
    ) -> UUID:
        if not user_id:
            raise ValueError("user_id is required to start session")
        
        if not restaurant_id:
            raise ValueError("restaurant_id is required to start session")
        
        if not meal_intent:
            raise ValueError("meal_intent is required to start session")
        
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        time_of_day = self._detect_time_of_day(hour)
        user_experience_level = self._get_user_experience_level(db_session, user_id)
        
        restaurant = db_session.get(Restaurant, restaurant_id)
        if not restaurant:
            raise ValueError(f"Restaurant {restaurant_id} not found")
        
        session = RecommendationSession(
            user_id=user_id,
            restaurant_id=restaurant_id,
            meal_intent=meal_intent,
            hunger_level=hunger_level,
            time_of_day=time_of_day,
            detected_hour=hour,
            day_of_week=day_of_week,
            budget=budget,
            time_constraint_minutes=time_constraint_minutes,
            mood=mood,
            occasion=occasion,
            dietary_notes=dietary_notes,
            user_experience_level=user_experience_level,
            status="active",
            started_at=current_time,
            iteration_count=0
        )
        
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)
        
        logger.info(
            "Session started",
            extra={
                "session_id": str(session.id),
                "user_id": str(user_id),
                "restaurant_id": str(restaurant_id),
                "meal_intent": meal_intent,
                "time_of_day": time_of_day,
                "user_experience_level": user_experience_level
            }
        )
        
        return session.id
    
    def get_session(self, db_session: Session, session_id: UUID) -> RecommendationSession:
        session = db_session.get(RecommendationSession, session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        return session
    
    def complete_session(
        self,
        db_session: Session,
        session_id: UUID,
        selected_item_ids: list[UUID]
    ) -> RecommendationSession:
        session = self.get_session(db_session, session_id)
        
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.selected_items = [str(item_id) for item_id in selected_item_ids]
        flag_modified(session, "selected_items")
        
        for item_id in selected_item_ids:
            order_history = UserOrderHistory(
                user_id=session.user_id,
                item_id=item_id,
                restaurant_id=session.restaurant_id,
                session_id=session_id
            )
            db_session.add(order_history)
        
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)
        
        from services.communication.email_followup_service import email_followup_service
        email_followup_service.schedule_post_meal_email(
            db_session=db_session,
            session_id=session_id,
            delay_minutes=60
        )
        
        logger.info(
            "Session completed and post-meal email scheduled",
            extra={
                "session_id": str(session_id),
                "selected_items": len(selected_item_ids),
                "note": "Profile learning already persisted via UnifiedFeedbackService during session"
            }
        )
        
        return session
    
    def abandon_session(
        self,
        db_session: Session,
        session_id: UUID
    ) -> RecommendationSession:
        session = self.get_session(db_session, session_id)
        
        session.status = "abandoned"
        session.completed_at = datetime.utcnow()
        
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)
        
        logger.info("Session abandoned", extra={"session_id": str(session_id)})
        
        return session
    
    def add_items_shown(
        self,
        db_session: Session,
        session_id: UUID,
        item_ids: list[UUID]
    ) -> None:
        session = self.get_session(db_session, session_id)
        
        existing = set(session.items_shown)
        new_items = [str(item_id) for item_id in item_ids if str(item_id) not in existing]
        
        if new_items:
            session.items_shown = session.items_shown + new_items
            flag_modified(session, "items_shown")
            session.iteration_count += 1
            
            db_session.add(session)
            db_session.commit()
            
            logger.info(
                "Items added to session shown list",
                extra={
                    "session_id": str(session_id),
                    "new_items_count": len(new_items),
                    "total_shown": len(session.items_shown)
                }
            )
    
    def add_excluded_item(
        self,
        db_session: Session,
        session_id: UUID,
        item_id: UUID
    ) -> None:
        session = self.get_session(db_session, session_id)
        
        if str(item_id) not in session.excluded_items:
            session.excluded_items = session.excluded_items + [str(item_id)]
            flag_modified(session, "excluded_items")
            db_session.add(session)
            db_session.commit()
            
            logger.info(
                "Item added to session exclusions",
                extra={
                    "session_id": str(session_id),
                    "item_id": str(item_id),
                    "total_excluded": len(session.excluded_items)
                }
            )
    
    def set_active_composition(
        self,
        db_session: Session,
        session_id: UUID,
        composition_id: str,
        appetizer_id: UUID,
        main_id: UUID,
        dessert_id: UUID
    ) -> None:
        """Set the currently active composition and initialize validation state"""
        session = self.get_session(db_session, session_id)
        
        session.active_composition_id = composition_id
        
        # Initialize validation state for this composition
        if not session.composition_validation_state:
            session.composition_validation_state = {}
        
        session.composition_validation_state[composition_id] = {
            "appetizer": {
                "item_id": str(appetizer_id),
                "status": "pending"
            },
            "main": {
                "item_id": str(main_id),
                "status": "pending"
            },
            "dessert": {
                "item_id": str(dessert_id),
                "status": "pending"
            }
        }
        flag_modified(session, "composition_validation_state")
        
        db_session.add(session)
        # Note: Caller must commit
        
        logger.info(
            "Active composition set",
            extra={
                "session_id": str(session_id),
                "composition_id": composition_id
            }
        )
    
    def update_composition_validation_state(
        self,
        db_session: Session,
        session_id: UUID,
        composition_id: str,
        validation_state: Dict
    ) -> None:
        """Update validation state after user provides feedback"""
        session = self.get_session(db_session, session_id)
        
        if not session.composition_validation_state:
            session.composition_validation_state = {}
        
        session.composition_validation_state[composition_id] = validation_state
        flag_modified(session, "composition_validation_state")
        
        db_session.add(session)
        # Note: Caller must commit
        
        logger.info(
            "Composition validation state updated",
            extra={
                "session_id": str(session_id),
                "composition_id": composition_id,
                "validation_state": validation_state
            }
        )
    
    def set_pending_partial_regeneration(
        self,
        db_session: Session,
        session_id: UUID,
        accepted: Dict,
        rejected: list
    ) -> None:
        """Store which courses are accepted/rejected for partial regeneration."""
        session = self.get_session(db_session, session_id)
        session.pending_partial_regen = {
            "accepted": accepted,
            "rejected": rejected,
        }
        flag_modified(session, "pending_partial_regen")
        db_session.add(session)

        logger.info(
            "Pending partial regeneration set",
            extra={
                "session_id": str(session_id),
                "accepted_courses": list(accepted.keys()),
                "rejected_courses": rejected,
            }
        )

    def clear_pending_partial_regeneration(
        self,
        db_session: Session,
        session_id: UUID,
    ) -> None:
        """Clear partial regeneration state (used for regenerate_all)."""
        session = self.get_session(db_session, session_id)
        session.pending_partial_regen = None
        flag_modified(session, "pending_partial_regen")
        db_session.add(session)

    def _detect_time_of_day(self, hour: int) -> str:
        if 6 <= hour < 11:
            return "morning"
        elif 11 <= hour < 15:
            return "afternoon"
        elif 15 <= hour < 18:
            return "late_afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _get_user_experience_level(self, db_session: Session, user_id: UUID) -> str:
        rating_count = db_session.exec(
            select(Rating).where(Rating.user_id == user_id)
        ).all()
        
        interaction_count = db_session.exec(
            select(Interaction).where(Interaction.user_id == user_id)
        ).all()
        
        total_feedback = len(rating_count) + len(interaction_count)
        
        if total_feedback <= 5:
            return "new"
        elif total_feedback <= 20:
            return "learning"
        else:
            return "established"
    
    def get_user_experience_level(self, db_session: Session, user_id: UUID) -> str:
        return self._get_user_experience_level(db_session, user_id)
    
    def get_restaurant_visit_history(
        self,
        db_session: Session,
        user_id: UUID,
        restaurant_id: UUID
    ) -> Dict:
        """
        Get user's visit history at a specific restaurant.
        Returns information about previous sessions and orders.
        """
        if not user_id:
            raise ValueError("user_id is required to get visit history")
        
        if not restaurant_id:
            raise ValueError("restaurant_id is required to get visit history")
        
        previous_sessions = db_session.exec(
            select(RecommendationSession)
            .where(RecommendationSession.user_id == user_id)
            .where(RecommendationSession.restaurant_id == restaurant_id)
            .where(RecommendationSession.status == "completed")
            .order_by(RecommendationSession.completed_at.desc())
        ).all()
        
        is_repeat_visit = len(previous_sessions) > 0
        previous_visit_count = len(previous_sessions)
        last_visit_date = previous_sessions[0].completed_at if previous_sessions else None
        
        favorite_items = []
        recent_sessions_data = []
        
        if is_repeat_visit:
            from models import MenuItem
            
            item_order_counts = {}
            item_last_ordered = {}
            item_ratings = {}
            
            for session in previous_sessions[:10]:
                session_data = {
                    "session_id": str(session.id),
                    "date": session.started_at,
                    "items_ordered": [str(item_id) for item_id in session.selected_items]
                }
                recent_sessions_data.append(session_data)
                
                for item_id in session.selected_items:
                    item_id_str = str(item_id)
                    item_order_counts[item_id_str] = item_order_counts.get(item_id_str, 0) + 1
                    
                    if item_id_str not in item_last_ordered or session.completed_at > item_last_ordered[item_id_str]:
                        item_last_ordered[item_id_str] = session.completed_at
            
            ratings = db_session.exec(
                select(Rating)
                .where(Rating.user_id == user_id)
            ).all()
            
            for rating in ratings:
                item_ratings[str(rating.item_id)] = rating.rating
            
            for item_id_str, count in sorted(item_order_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                try:
                    from uuid import UUID as UUIDType
                    menu_item = db_session.get(MenuItem, UUIDType(item_id_str))
                    if menu_item:
                        favorite_items.append({
                            "item_id": item_id_str,
                            "name": menu_item.name,
                            "times_ordered": count,
                            "last_ordered": item_last_ordered.get(item_id_str),
                            "rating": item_ratings.get(item_id_str)
                        })
                except Exception as e:
                    logger.warning(f"Could not fetch menu item {item_id_str}", extra={"error": str(e)})
                    continue
        
        result = {
            "is_repeat_visit": is_repeat_visit,
            "previous_visit_count": previous_visit_count,
            "last_visit_date": last_visit_date,
            "favorite_items": favorite_items,
            "recent_sessions": recent_sessions_data
        }
        
        logger.info(
            "Restaurant visit history retrieved",
            extra={
                "user_id": str(user_id),
                "restaurant_id": str(restaurant_id),
                "is_repeat_visit": is_repeat_visit,
                "visit_count": previous_visit_count
            }
        )
        
        return result
