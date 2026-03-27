from datetime import datetime, timedelta
from typing import Optional
import jwt
import secrets
from sqlmodel import Session, select
from models.user import User
from models.auth import OTPCode, UserSession
from models.access_request import AccessRequest
from services.communication.email_service import email_service
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AuthenticationError(Exception):
    pass


class AuthService:
    @staticmethod
    def check_early_access(email: str, db: Session) -> str:
        """Returns 'allowed', 'pending', 'denied', or 'unknown'."""
        if not settings.EARLY_ACCESS_ENABLED:
            return "allowed"

        email = email.lower().strip()

        if settings.EARLY_ACCESS_EXEMPT_EMAIL and email == settings.EARLY_ACCESS_EXEMPT_EMAIL.lower().strip():
            return "allowed"

        access_req = db.exec(
            select(AccessRequest).where(AccessRequest.email == email)
        ).first()

        if access_req is None:
            return "unknown"
        if access_req.status == "approved":
            return "allowed"
        if access_req.status == "pending":
            return "pending"
        if access_req.status == "denied":
            return "denied"

        return "unknown"

    @staticmethod
    def request_otp(email: str, db: Session):
        if not email or "@" not in email:
            raise ValueError("email must be a valid email address")

        email = email.lower().strip()

        # Early-access gate (no-op when EARLY_ACCESS_ENABLED=False)
        access_status = AuthService.check_early_access(email, db)
        if access_status != "allowed":
            return access_status

        user = db.exec(select(User).where(User.email == email)).first()
        if not user:
            user = User(email=email, email_verified=False, onboarding_completed=False)
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info("New user created via OTP request", extra={"email": email, "user_id": str(user.id)})
        
        existing_otp = db.exec(
            select(OTPCode).where(
                OTPCode.user_id == user.id,
                OTPCode.is_used == False
            )
        ).first()
        
        if existing_otp:
            if existing_otp.is_valid():
                email_sent = email_service.send_otp_code(user.email, existing_otp.code)
                if not email_sent:
                    raise AuthenticationError("Failed to send OTP email")
                return existing_otp
            else:
                existing_otp.is_used = True
                db.add(existing_otp)
        
        otp = OTPCode(
            user_id=user.id,
            code=OTPCode.generate_code(),
            expires_at=datetime.utcnow() + timedelta(minutes=settings.OTP_EXPIRE_MINUTES)
        )
        db.add(otp)
        db.commit()
        db.refresh(otp)
        
        email_sent = email_service.send_otp_code(user.email, otp.code)
        if not email_sent:
            raise AuthenticationError("Failed to send OTP email")
        
        logger.info("OTP generated and sent", extra={"user_id": str(user.id), "email": email})
        return otp
    
    @staticmethod
    def verify_otp(email: str, code: str, device_info: Optional[str], db: Session) -> UserSession:
        if not email:
            raise ValueError("email is required")
        
        if not code:
            raise ValueError("code is required")
        
        email = email.lower().strip()
        code = code.strip()
        
        user = db.exec(select(User).where(User.email == email)).first()
        if not user:
            raise AuthenticationError("Invalid email or code")
        
        otp = db.exec(
            select(OTPCode).where(
                OTPCode.user_id == user.id,
                OTPCode.code == code,
                OTPCode.is_used == False
            )
        ).first()
        
        if not otp:
            raise AuthenticationError("Invalid email or code")
        
        if not otp.is_valid():
            otp.is_used = True
            db.add(otp)
            db.commit()
            raise AuthenticationError("OTP code has expired or exceeded max attempts")
        
        otp.attempts += 1
        
        if otp.attempts >= settings.OTP_MAX_ATTEMPTS:
            otp.is_used = True
            db.add(otp)
            db.commit()
            raise AuthenticationError("Maximum OTP attempts exceeded")
        
        otp.is_used = True
        db.add(otp)
        
        if not user.email_verified:
            user.email_verified = True
            db.add(user)
        
        user.last_login = datetime.utcnow()
        db.add(user)
        
        access_token = AuthService._create_access_token(user.id)
        refresh_token = AuthService._create_refresh_token(user.id)
        
        session = UserSession(
            user_id=user.id,
            token=access_token,
            refresh_token=refresh_token,
            device_info=device_info,
            expires_at=datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Cache email in Redis for rate limiter lookup
        try:
            from utils.redis_client import get_redis
            r = get_redis()
            if r:
                r.setex(
                    f"token_email:{access_token}",
                    settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                    email,
                )
        except Exception:
            pass  # fail open — rate limiter falls back to IP

        logger.info("OTP verified successfully", extra={"user_id": str(user.id), "email": email})
        return session
    
    @staticmethod
    def refresh_session(refresh_token: str, db: Session) -> UserSession:
        if not refresh_token:
            raise ValueError("refresh_token is required")
        
        try:
            payload = jwt.decode(
                refresh_token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            token_type = payload.get("type")
            
            if token_type != "refresh":
                raise AuthenticationError("Invalid token type")
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Refresh token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid refresh token")
        
        session = db.exec(
            select(UserSession).where(UserSession.refresh_token == refresh_token)
        ).first()
        
        if not session:
            raise AuthenticationError("Session not found")
        
        user = db.exec(select(User).where(User.id == user_id)).first()
        if not user:
            raise AuthenticationError("User not found")
        
        new_access_token = AuthService._create_access_token(user.id)
        
        session.token = new_access_token
        session.expires_at = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        session.last_used_at = datetime.utcnow()
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        logger.info("Session refreshed", extra={"user_id": str(user.id)})
        return session
    
    @staticmethod
    def verify_token(token: str, db: Session) -> User:
        if not token:
            raise ValueError("token is required")
        
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM]
            )
            user_id = payload.get("sub")
            token_type = payload.get("type")
            
            if token_type != "access":
                raise AuthenticationError("Invalid token type")
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Access token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid access token")
        
        user = db.exec(select(User).where(User.id == user_id)).first()
        if not user:
            raise AuthenticationError("User not found")
        
        return user
    
    @staticmethod
    def logout(token: str, db: Session) -> None:
        if not token:
            raise ValueError("token is required")
        
        session = db.exec(select(UserSession).where(UserSession.token == token)).first()
        if session:
            db.delete(session)
            db.commit()
            logger.info("User logged out", extra={"user_id": str(session.user_id)})
    
    @staticmethod
    def _create_access_token(user_id: str) -> str:
        expire = datetime.utcnow() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": str(user_id),
            "type": "access",
            "exp": expire
        }
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    @staticmethod
    def _create_refresh_token(user_id: str) -> str:
        expire = datetime.utcnow() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = {
            "sub": str(user_id),
            "type": "refresh",
            "exp": expire
        }
        return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


auth_service = AuthService()
