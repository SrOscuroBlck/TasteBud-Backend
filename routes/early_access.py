from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlmodel import Session
from config.database import get_session
from config.settings import settings
from services.communication.early_access_service import early_access_service
from utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()


class AccessRequestBody(BaseModel):
    email: str


@router.post("/early-access/request")
def request_access(body: AccessRequestBody, session: Session = Depends(get_session)):
    if not settings.EARLY_ACCESS_ENABLED:
        return JSONResponse(status_code=400, content={"status": "not_active", "message": "Early access is not active."})

    try:
        early_access_service.create_access_request(body.email, session)
        return JSONResponse(status_code=201, content={"status": "access_requested", "message": "Your access request has been submitted."})
    except ValueError as e:
        err = str(e)
        if err == "already_requested":
            return JSONResponse(status_code=409, content={"status": "already_requested", "message": "An access request for this email already exists."})
        if err == "invalid_email":
            return JSONResponse(status_code=400, content={"status": "invalid_email", "message": "We couldn't deliver an email to this address. Please check and try again."})
        raise
    except Exception as e:
        logger.error("Error creating access request", extra={"error": str(e)}, exc_info=True)
        return JSONResponse(status_code=500, content={"status": "error", "message": "Internal server error"})


@router.get("/early-access/approve")
def approve_access(token: str = Query(...), session: Session = Depends(get_session)):
    try:
        early_access_service.resolve_request(token, "approve", session)
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/early-access/confirmed?status=approved",
            status_code=302,
        )
    except ValueError:
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/early-access/confirmed?status=error",
            status_code=302,
        )


@router.get("/early-access/deny")
def deny_access(token: str = Query(...), session: Session = Depends(get_session)):
    try:
        early_access_service.resolve_request(token, "deny", session)
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/early-access/confirmed?status=denied",
            status_code=302,
        )
    except ValueError:
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/early-access/confirmed?status=error",
            status_code=302,
        )
