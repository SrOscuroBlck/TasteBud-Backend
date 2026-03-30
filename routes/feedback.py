from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from datetime import datetime
from config.database import get_session
from models import User
from models.session import PostMealFeedback, RecommendationSession
from routes.api import get_current_user
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/feedback", tags=["post-meal-feedback"])


class PostMealFeedbackRequest(BaseModel):
    session_id: UUID
    items_ordered: List[UUID] = Field(..., min_items=1)
    overall_satisfaction: int = Field(..., ge=1, le=5)
    would_order_again: bool
    taste_match: int = Field(..., ge=1, le=5)
    portion_size_rating: Optional[int] = Field(None, ge=1, le=5)
    value_for_money: Optional[int] = Field(None, ge=1, le=5)
    service_quality: Optional[int] = Field(None, ge=1, le=5)
    wait_time_minutes: Optional[int] = None
    additional_notes: Optional[str] = None


class SubmitFeedbackByTokenRequest(BaseModel):
    overall_satisfaction: int = Field(..., ge=1, le=5)
    would_order_again: bool
    taste_match: int = Field(..., ge=1, le=5)
    portion_size_rating: Optional[int] = Field(None, ge=1, le=5)
    value_for_money: Optional[int] = Field(None, ge=1, le=5)
    service_quality: Optional[int] = Field(None, ge=1, le=5)
    wait_time_minutes: Optional[int] = None
    additional_notes: Optional[str] = None


@router.get("/pending")
def get_pending_feedback(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    pending_sessions = session.exec(
        select(RecommendationSession)
        .where(RecommendationSession.user_id == current_user.id)
        .where(RecommendationSession.status == "completed")
    ).all()
    
    pending_feedback = []
    for rec_session in pending_sessions:
        existing_feedback = session.exec(
            select(PostMealFeedback)
            .where(PostMealFeedback.session_id == rec_session.id)
        ).first()
        
        if not existing_feedback:
            pending_feedback.append({
                "session_id": str(rec_session.id),
                "restaurant_id": str(rec_session.restaurant_id),
                "meal_intent": rec_session.meal_intent.value if hasattr(rec_session.meal_intent, 'value') else rec_session.meal_intent,
                "completed_at": rec_session.completed_at.isoformat() if rec_session.completed_at else None
            })
    
    logger.info(
        "Pending feedback retrieved",
        extra={"user_id": str(current_user.id), "count": len(pending_feedback)}
    )
    
    return {"pending_feedback": pending_feedback, "count": len(pending_feedback)}


@router.post("/post-meal")
def submit_post_meal_feedback(
    request: PostMealFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required for post-meal feedback")
    
    if not request.items_ordered or len(request.items_ordered) == 0:
        raise HTTPException(status_code=400, detail="items_ordered list cannot be empty")
    
    if request.overall_satisfaction < 1 or request.overall_satisfaction > 5:
        raise HTTPException(status_code=400, detail="overall_satisfaction must be between 1 and 5")
    
    if request.taste_match < 1 or request.taste_match > 5:
        raise HTTPException(status_code=400, detail="taste_match must be between 1 and 5")
    
    rec_session = db_session.get(RecommendationSession, request.session_id)
    
    if not rec_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if rec_session.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot submit feedback for other user's session")
    
    if rec_session.status != "completed":
        raise HTTPException(status_code=400, detail="Can only submit feedback for completed sessions")
    
    existing_feedback = db_session.exec(
        select(PostMealFeedback)
        .where(PostMealFeedback.session_id == request.session_id)
    ).first()
    
    if existing_feedback:
        raise HTTPException(status_code=400, detail="Feedback already submitted for this session")
    
    feedback = PostMealFeedback(
        session_id=request.session_id,
        items_ordered=request.items_ordered,
        overall_satisfaction=request.overall_satisfaction,
        would_order_again=request.would_order_again,
        taste_match=request.taste_match,
        portion_size_rating=request.portion_size_rating,
        value_for_money=request.value_for_money,
        service_quality=request.service_quality,
        wait_time_minutes=request.wait_time_minutes,
        additional_notes=request.additional_notes
    )
    
    db_session.add(feedback)
    db_session.commit()
    
    from services.learning.online_learning_service import OnlineLearningService
    
    learning_service = OnlineLearningService()
    learning_service.update_from_post_meal_feedback(
        db_session=db_session,
        user=current_user,
        feedback=feedback,
        session=rec_session
    )
    
    logger.info(
        "Post-meal feedback submitted",
        extra={
            "user_id": str(current_user.id),
            "session_id": str(request.session_id),
            "overall_satisfaction": request.overall_satisfaction,
            "taste_match": request.taste_match
        }
    )
    
    return {
        "success": True,
        "message": "Thank you for your feedback! Your taste profile has been updated."
    }


@router.get("/submit/{token}")
def get_feedback_form(
    token: str,
    session: Session = Depends(get_session)
):
    from services.communication.email_service import email_service
    
    session_id = email_service.verify_feedback_token(token)
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid or expired feedback token")
    
    rec_session = session.get(RecommendationSession, session_id)
    
    if not rec_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    existing_feedback = session.exec(
        select(PostMealFeedback)
        .where(PostMealFeedback.session_id == session_id)
    ).first()
    
    if existing_feedback:
        return {
            "already_submitted": True,
            "message": "Feedback already submitted for this meal. Thank you!"
        }
    
    from models import MenuItem
    
    selected_items = []
    for item_id in rec_session.selected_items:
        item = session.get(MenuItem, item_id)
        if item:
            selected_items.append({
                "id": str(item.id),
                "name": item.name,
                "description": item.description
            })
    
    return {
        "already_submitted": False,
        "session_id": str(rec_session.id),
        "meal_intent": rec_session.meal_intent.value if hasattr(rec_session.meal_intent, 'value') else rec_session.meal_intent,
        "selected_items": selected_items,
        "completed_at": rec_session.completed_at.isoformat() if rec_session.completed_at else None
    }


@router.post("/submit/{token}")
def submit_feedback_by_token(
    token: str,
    request: SubmitFeedbackByTokenRequest,
    session: Session = Depends(get_session)
):
    if request.overall_satisfaction < 1 or request.overall_satisfaction > 5:
        raise HTTPException(status_code=400, detail="overall_satisfaction must be between 1 and 5")
    
    if request.taste_match < 1 or request.taste_match > 5:
        raise HTTPException(status_code=400, detail="taste_match must be between 1 and 5")
    
    from services.communication.email_service import email_service
    
    session_id = email_service.verify_feedback_token(token)
    
    if not session_id:
        raise HTTPException(status_code=400, detail="Invalid or expired feedback token")
    
    rec_session = session.get(RecommendationSession, session_id)
    
    if not rec_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    existing_feedback = session.exec(
        select(PostMealFeedback)
        .where(PostMealFeedback.session_id == session_id)
    ).first()
    
    if existing_feedback:
        raise HTTPException(status_code=400, detail="Feedback already submitted for this session")
    
    feedback = PostMealFeedback(
        session_id=session_id,
        items_ordered=rec_session.selected_items,
        overall_satisfaction=request.overall_satisfaction,
        would_order_again=request.would_order_again,
        taste_match=request.taste_match,
        portion_size_rating=request.portion_size_rating,
        value_for_money=request.value_for_money,
        service_quality=request.service_quality,
        wait_time_minutes=request.wait_time_minutes,
        additional_notes=request.additional_notes
    )
    
    session.add(feedback)
    session.commit()
    
    user = session.get(User, rec_session.user_id)
    
    from services.learning.online_learning_service import OnlineLearningService
    
    learning_service = OnlineLearningService()
    learning_service.update_from_post_meal_feedback(
        db_session=session,
        user=user,
        feedback=feedback,
        session=rec_session
    )
    
    logger.info(
        "Post-meal feedback submitted via token",
        extra={
            "session_id": str(session_id),
            "overall_satisfaction": request.overall_satisfaction,
            "taste_match": request.taste_match
        }
    )
    
    return {
        "success": True,
        "message": "Thank you for your feedback! Your taste profile has been updated."
    }
