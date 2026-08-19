"""
ORM models: User, Message, Feedback.
"""
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class User(Base):
    __tablename__ = "users"

    id:            Mapped[int]      = mapped_column(Integer, primary_key=True)
    email:         Mapped[str]      = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str]      = mapped_column(String(255))
    is_admin:      Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Message(Base):
    """One chat exchange: user prompt + model reply."""
    __tablename__ = "messages"

    id:          Mapped[int]      = mapped_column(Integer, primary_key=True)
    session_id:  Mapped[str]      = mapped_column(String(64), index=True)
    prompt:      Mapped[str]      = mapped_column(Text)
    reply:       Mapped[str]      = mapped_column(Text)
    temperature: Mapped[float]    = mapped_column(Float)
    top_k:       Mapped[int]      = mapped_column(Integer)
    max_tokens:  Mapped[int]      = mapped_column(Integer)
    created_at:  Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    feedback: Mapped[list["Feedback"]] = relationship(back_populates="message",
                                                      cascade="all, delete-orphan")


class Feedback(Base):
    """Admin feedback attached to a message. Rating scale matches HybridTeacher."""
    __tablename__ = "feedback"

    id:             Mapped[int]      = mapped_column(Integer, primary_key=True)
    message_id:     Mapped[int]      = mapped_column(ForeignKey("messages.id"), index=True)
    rating:         Mapped[str]      = mapped_column(String(8))   # +++ ++ + = -
    corrected_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    author_id:      Mapped[int]      = mapped_column(ForeignKey("users.id"))
    created_at:     Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at:     Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    message: Mapped[Message] = relationship(back_populates="feedback")


VALID_RATINGS = ("+++", "++", "+", "=", "-")
RATING_WEIGHT = {"+++": 1.0, "++": 0.8, "+": 0.5, "=": 0.0, "-": -0.8}
