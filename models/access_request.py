from datetime import datetime
from typing import Optional
from uuid import uuid4, UUID
from sqlmodel import SQLModel, Field


class AccessRequest(SQLModel, table=True):
    __tablename__ = "access_request"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    email: str = Field(unique=True, index=True)
    status: str = Field(default="pending")  # "pending" | "approved" | "denied"
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(default=None)
    admin_token: str = Field(index=True)
    token_used: bool = Field(default=False)
    notes: Optional[str] = Field(default=None)
