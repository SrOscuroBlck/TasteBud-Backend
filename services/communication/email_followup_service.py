from __future__ import annotations
from typing import List, Optional
from uuid import UUID
from datetime import datetime, timedelta
from sqlmodel import Session, select
import secrets
import json
from models import User, MenuItem
from models.session import RecommendationSession
from services.communication.email_service import email_service
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailFollowUpService:
    
    def __init__(self):
        self.feedback_tokens = {}
    
    def schedule_post_meal_email(
        self,
        db_session: Session,
        session_id: UUID,
        delay_minutes: int = 60
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required to schedule post-meal email")
        
        rec_session = db_session.get(RecommendationSession, session_id)
        
        if not rec_session:
            raise ValueError(f"Session {session_id} not found")
        
        user = db_session.get(User, rec_session.user_id)
        
        if not user or not user.email:
            logger.warning(
                "Cannot schedule email for user without email",
                extra={"user_id": str(rec_session.user_id)}
            )
            return
        
        token = self._generate_feedback_token(session_id)
        
        selected_items = []
        for item_id in rec_session.selected_items:
            item = db_session.get(MenuItem, item_id)
            if item:
                selected_items.append({
                    "id": str(item.id),
                    "name": item.name,
                    "description": item.description,
                    "price": item.price
                })
        
        context = {
            "user_name": user.email.split("@")[0],
            "meal_intent": rec_session.meal_intent,
            "selected_items": selected_items,
            "feedback_link": self._generate_feedback_link(token),
            "token": token
        }
        
        scheduled_time = datetime.utcnow() + timedelta(minutes=delay_minutes)
        
        rec_session.email_scheduled_at = scheduled_time
        db_session.add(rec_session)
        db_session.commit()
        
        logger.info(
            "Post-meal email scheduled",
            extra={
                "session_id": str(session_id),
                "user_email": user.email,
                "scheduled_time": scheduled_time.isoformat(),
                "delay_minutes": delay_minutes
            }
        )
        
        try:
            self._send_post_meal_email_immediately(user.email, context)
            logger.info(
                "Post-meal email sent immediately (background scheduler not implemented)",
                extra={"session_id": str(session_id)}
            )
        except Exception as e:
            logger.error(
                "Failed to send post-meal email",
                extra={"error": str(e), "session_id": str(session_id)},
                exc_info=True
            )
    
    def _generate_feedback_token(self, session_id: UUID) -> str:
        token = secrets.token_urlsafe(32)
        
        self.feedback_tokens[token] = {
            "session_id": session_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(days=7)
        }
        
        return token
    
    def verify_feedback_token(self, token: str) -> Optional[UUID]:
        if not token:
            return None
        
        token_data = self.feedback_tokens.get(token)
        
        if not token_data:
            return None
        
        if datetime.utcnow() > token_data["expires_at"]:
            del self.feedback_tokens[token]
            return None
        
        return token_data["session_id"]
    
    def _generate_feedback_link(self, token: str) -> str:
        return f"{settings.BACKEND_BASE_URL}/api/v1/feedback/submit/{token}"
    
    def _send_post_meal_email_immediately(self, recipient_email: str, context: dict) -> None:
        subject = "How was your meal? ️"
        
        html_body = self._build_email_html(context)
        text_body = self._build_email_text(context)
        
        email_service.send_email(
            to_email=recipient_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body
        )
    
    def _build_email_html(self, context: dict) -> str:
        items_html = ""
        for item in context["selected_items"]:
            items_html += f"""
            <li style="margin-bottom: 10px;">
                <strong>{item['name']}</strong>
                {f"<br><span style='color: #666;'>{item['description']}</span>" if item.get('description') else ""}
            </li>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #FF6B35; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f9f9f9; padding: 30px; border-radius: 0 0 5px 5px; }}
                .button {{ display: inline-block; background-color: #FF6B35; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>️ TasteBud</h1>
                    <p>We'd love your feedback!</p>
                </div>
                <div class="content">
                    <p>Hi {context['user_name']},</p>
                    
                    <p>We hope you enjoyed your {context['meal_intent']} experience! Your feedback helps us provide better recommendations tailored to your taste.</p>
                    
                    <p><strong>You ordered:</strong></p>
                    <ul>
                        {items_html}
                    </ul>
                    
                    <p>Please take a moment to share your thoughts:</p>
                    
                    <div style="text-align: center;">
                        <a href="{context['feedback_link']}" class="button">Share Your Feedback</a>
                    </div>
                    
                    <p style="margin-top: 30px; font-size: 14px; color: #666;">
                        Your feedback is valuable and helps us learn your preferences better. It only takes a minute!
                    </p>
                </div>
                <div class="footer">
                    <p>TasteBud - Personalized Food Recommendations</p>
                    <p>This link expires in 7 days</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _build_email_text(self, context: dict) -> str:
        items_text = "\n".join([
            f"  - {item['name']}" + (f": {item['description']}" if item.get('description') else "")
            for item in context["selected_items"]
        ])
        
        text = f"""
Hi {context['user_name']},

We hope you enjoyed your {context['meal_intent']} experience! Your feedback helps us provide better recommendations tailored to your taste.

You ordered:
{items_text}

Please take a moment to share your thoughts by clicking the link below:

{context['feedback_link']}

Your feedback is valuable and helps us learn your preferences better. It only takes a minute!

---
TasteBud - Personalized Food Recommendations
This link expires in 7 days
        """
        
        return text.strip()
    
    def send_pending_emails(self, db_session: Session) -> int:
        now = datetime.utcnow()
        
        pending_sessions = db_session.exec(
            select(RecommendationSession)
            .where(RecommendationSession.status == "completed")
            .where(RecommendationSession.email_scheduled_at <= now)
            .where(RecommendationSession.email_sent_at.is_(None))
        ).all()
        
        sent_count = 0
        
        for rec_session in pending_sessions:
            try:
                user = db_session.get(User, rec_session.user_id)
                
                if not user or not user.email:
                    continue
                
                token = self._generate_feedback_token(rec_session.id)
                
                selected_items = []
                for item_id in rec_session.selected_items:
                    item = db_session.get(MenuItem, item_id)
                    if item:
                        selected_items.append({
                            "id": str(item.id),
                            "name": item.name,
                            "description": item.description,
                            "price": item.price
                        })
                
                context = {
                    "user_name": user.email.split("@")[0],
                    "meal_intent": rec_session.meal_intent.value,
                    "selected_items": selected_items,
                    "feedback_link": self._generate_feedback_link(token),
                    "token": token
                }
                
                self._send_post_meal_email_immediately(user.email, context)
                
                rec_session.email_sent_at = datetime.utcnow()
                db_session.add(rec_session)
                
                sent_count += 1
                
            except Exception as e:
                logger.error(
                    "Failed to send pending email",
                    extra={"error": str(e), "session_id": str(rec_session.id)},
                    exc_info=True
                )
        
        db_session.commit()
        
        logger.info(
            "Pending emails processed",
            extra={"sent_count": sent_count, "total_pending": len(pending_sessions)}
        )
        
        return sent_count


email_followup_service = EmailFollowUpService()
