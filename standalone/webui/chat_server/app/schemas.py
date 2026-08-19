"""
Pydantic DTOs for request/response bodies.
Envelope: {"success": bool, "data": T | null, "error": {...} | null}.
"""
from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T")


class ApiError(BaseModel):
    code: str
    message: str


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    data: Optional[T] = None
    error: Optional[ApiError] = None


# ---------------- Auth ----------------

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=255)
    password: str = Field(..., min_length=1, max_length=255)


class TokenPayload(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserPublic"


class UserPublic(BaseModel):
    id: int
    email: str
    is_admin: bool


# ---------------- Chat ----------------

class ChatSendRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(None, max_length=64)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1, le=500)
    max_tokens: Optional[int] = Field(None, ge=1, le=512)


class FeedbackPublic(BaseModel):
    id: int
    rating: str
    corrected_text: Optional[str] = None
    applied_at: Optional[datetime] = None
    created_at: datetime


class MessagePublic(BaseModel):
    id: int
    session_id: str
    prompt: str
    reply: str
    created_at: datetime
    feedback: list[FeedbackPublic] = []


class ChatSendResponse(BaseModel):
    message: MessagePublic


# ---------------- Feedback ----------------

class FeedbackCreateRequest(BaseModel):
    message_id: int
    rating: str = Field(..., pattern=r"^(\+\+\+|\+\+|\+|=|-)$")
    corrected_text: Optional[str] = Field(None, max_length=8000)


class FeedbackUpdateRequest(BaseModel):
    rating: Optional[str] = Field(None, pattern=r"^(\+\+\+|\+\+|\+|=|-)$")
    corrected_text: Optional[str] = Field(None, max_length=8000)


# ---------------- Admin ----------------

class BackupResponse(BaseModel):
    path: str
    size_bytes: int
    created_at: datetime


class TrainStatus(BaseModel):
    state: str            # idle | running | done | error
    progress: float = 0.0
    total: int = 0
    processed: int = 0
    last_loss: Optional[float] = None
    backup_path: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None


class TrainStartRequest(BaseModel):
    lr: Optional[float] = Field(None, ge=1e-7, le=1e-2)
    steps_per_sample: Optional[int] = Field(None, ge=1, le=50)


TokenPayload.model_rebuild()
