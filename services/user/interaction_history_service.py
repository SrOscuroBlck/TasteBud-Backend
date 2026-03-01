"""
Interaction History Service

Manages tracking of item views and interactions across sessions.
Supports novelty scoring and prevents showing same items repeatedly.
"""

from datetime import datetime
from uuid import UUID
from sqlmodel import Session, select
from typing import Optional, Dict
from models.interaction_history import UserItemInteractionHistory
from utils.logger import setup_logger

logger = setup_logger(__name__)


class InteractionHistoryService:
    
    def record_item_shown(
        self,
        db_session: Session,
        user_id: UUID,
        item_id: UUID,
        session_id: UUID
    ) -> UserItemInteractionHistory:
        """
        Record that an item was shown to a user in a session.
        Creates new record or updates existing one.
        """
        if not user_id:
            raise ValueError("user_id is required to record item shown")
        
        if not item_id:
            raise ValueError("item_id is required to record item shown")
        
        if not session_id:
            raise ValueError("session_id is required to record item shown")
        
        existing = db_session.exec(
            select(UserItemInteractionHistory)
            .where(UserItemInteractionHistory.user_id == user_id)
            .where(UserItemInteractionHistory.item_id == item_id)
        ).first()
        
        if existing:
            existing.last_shown_at = datetime.utcnow()
            existing.times_shown += 1
            
            session_id_str = str(session_id)
            if session_id_str not in existing.session_ids:
                existing.session_ids.append(session_id_str)
            
            db_session.add(existing)
            db_session.commit()
            db_session.refresh(existing)
            
            logger.debug(
                "Item view updated",
                extra={
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "times_shown": existing.times_shown
                }
            )
            
            return existing
        else:
            new_history = UserItemInteractionHistory(
                user_id=user_id,
                item_id=item_id,
                first_shown_at=datetime.utcnow(),
                last_shown_at=datetime.utcnow(),
                times_shown=1,
                session_ids=[str(session_id)]
            )
            
            db_session.add(new_history)
            db_session.commit()
            db_session.refresh(new_history)
            
            logger.debug(
                "Item view recorded",
                extra={
                    "user_id": str(user_id),
                    "item_id": str(item_id)
                }
            )
            
            return new_history
    
    def get_user_item_history(
        self,
        db_session: Session,
        user_id: UUID,
        item_id: UUID
    ) -> Optional[UserItemInteractionHistory]:
        """Get interaction history for a specific user-item pair"""
        if not user_id or not item_id:
            return None
        
        return db_session.exec(
            select(UserItemInteractionHistory)
            .where(UserItemInteractionHistory.user_id == user_id)
            .where(UserItemInteractionHistory.item_id == item_id)
        ).first()
    
    def get_all_user_history(
        self,
        db_session: Session,
        user_id: UUID
    ) -> Dict[UUID, UserItemInteractionHistory]:
        """Get all interaction history for a user, keyed by item_id"""
        if not user_id:
            return {}
        
        histories = db_session.exec(
            select(UserItemInteractionHistory)
            .where(UserItemInteractionHistory.user_id == user_id)
        ).all()
        
        return {history.item_id: history for history in histories}
    
    def update_interaction_outcome(
        self,
        db_session: Session,
        user_id: UUID,
        item_id: UUID,
        was_dismissed: bool = False,
        was_disliked: bool = False,
        was_liked: bool = False,
        was_ordered: bool = False
    ) -> None:
        """
        Update the outcome of an interaction.
        Called when user gives feedback on an item.
        """
        if not user_id or not item_id:
            raise ValueError("user_id and item_id are required to update outcome")
        
        history = self.get_user_item_history(db_session, user_id, item_id)
        
        if history:
            if was_dismissed:
                history.was_dismissed = True
            if was_disliked:
                history.was_disliked = True
            if was_liked:
                history.was_liked = True
            if was_ordered:
                history.was_ordered = True
            
            db_session.add(history)
            db_session.commit()
            
            logger.debug(
                "Interaction outcome updated",
                extra={
                    "user_id": str(user_id),
                    "item_id": str(item_id),
                    "was_ordered": was_ordered,
                    "was_liked": was_liked
                }
            )
    
    def calculate_novelty_bonus(
        self,
        user_history: Optional[UserItemInteractionHistory]
    ) -> float:
        """
        Calculate novelty bonus/penalty for an item based on interaction history.
        
        Returns:
        - Never seen: +0.3
        - Previously disliked/skipped: -0.8 (STRONG PENALTY)
        - Ordered and liked: +0.1 (familiarity bonus)
        - Seen 1-2 times: +0.1
        - Seen 3-5 times: 0.0
        - Seen 6+ times: -0.2
        """
        if not user_history:
            return 0.3
        
        if user_history.was_disliked:
            return -0.8
        
        if user_history.was_dismissed:
            return -0.3
        
        if user_history.was_ordered and user_history.was_liked:
            return 0.1
        
        times_shown = user_history.times_shown
        
        if times_shown >= 6:
            return -0.2
        elif times_shown >= 3:
            return 0.0
        else:
            return 0.1
