import hmac
import hashlib
import dns.resolver
import re
from datetime import datetime
from uuid import uuid4
from sqlmodel import Session, select
from models.access_request import AccessRequest
from models.user import User
from services.communication.email_service import email_service
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


class EarlyAccessService:

    @staticmethod
    def _generate_admin_token(email: str) -> str:
        message = f"{email}:{uuid4()}"
        return hmac.new(
            settings.EARLY_ACCESS_TOKEN_SECRET.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _validate_email_domain(email: str) -> bool:
        """Check that the email domain has valid MX records."""
        domain = email.split("@")[1]
        try:
            mx_records = dns.resolver.resolve(domain, "MX")
            return len(mx_records) > 0
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
            return False
        except Exception as e:
            logger.warning(f"DNS MX lookup failed for {domain}: {e}")
            return False

    @staticmethod
    def create_access_request(email: str, db: Session) -> AccessRequest:
        email = email.lower().strip()

        if not EMAIL_RE.match(email):
            raise ValueError("invalid_email")

        if not EarlyAccessService._validate_email_domain(email):
            raise ValueError("invalid_email")

        existing = db.exec(
            select(AccessRequest).where(AccessRequest.email == email)
        ).first()

        if existing:
            raise ValueError("already_requested")

        # Send confirmation to requester — validates deliverability
        confirmation_sent = email_service.send_access_request_confirmation(email)
        if not confirmation_sent:
            raise ValueError("invalid_email")

        req = AccessRequest(
            email=email,
            admin_token=EarlyAccessService._generate_admin_token(email),
        )
        db.add(req)
        db.commit()
        db.refresh(req)

        # Notify admin with approve/deny links
        email_service.send_access_request_notification(
            requester_email=email,
            token=req.admin_token,
        )

        logger.info("Access request created", extra={"email": email, "request_id": str(req.id)})
        return req

    @staticmethod
    def resolve_request(token: str, action: str, db: Session) -> AccessRequest:
        req = db.exec(
            select(AccessRequest).where(
                AccessRequest.admin_token == token,
                AccessRequest.token_used == False,
            )
        ).first()

        if not req:
            raise ValueError("Invalid or already-used token")

        req.status = "approved" if action == "approve" else "denied"
        req.resolved_at = datetime.utcnow()
        req.token_used = True
        db.add(req)

        if action == "approve":
            user = db.exec(select(User).where(User.email == req.email)).first()
            if not user:
                user = User(email=req.email, email_verified=False, onboarding_completed=False)
                db.add(user)

        db.commit()
        db.refresh(req)

        logger.info(f"Access request {action}d", extra={"email": req.email, "request_id": str(req.id)})
        return req


early_access_service = EarlyAccessService()
