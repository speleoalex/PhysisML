"""
Chat endpoints — anonymous (no auth required for send/history).
"""
import secrets
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import inference
from app.config import settings
from app.db import get_db
from app.models import Message
from app.schemas import (ApiResponse, ChatSendRequest, ChatSendResponse,
                         FeedbackPublic, MessagePublic)

router = APIRouter(prefix="/chat", tags=["chat"])


def _to_public(m: Message, feedback: list | None = None) -> MessagePublic:
    """
    Build the public view of a Message.
    `feedback` must be passed explicitly (pre-loaded) — accessing
    `m.feedback` directly would trigger lazy I/O on an async session.
    """
    fb_items = feedback if feedback is not None else []
    return MessagePublic(
        id=m.id,
        session_id=m.session_id,
        prompt=m.prompt,
        reply=m.reply,
        created_at=m.created_at,
        feedback=[
            FeedbackPublic(
                id=f.id,
                rating=f.rating,
                corrected_text=f.corrected_text,
                applied_at=f.applied_at,
                created_at=f.created_at,
            )
            for f in fb_items
        ],
    )


@router.post("/send", response_model=ApiResponse[ChatSendResponse])
async def send(body: ChatSendRequest, db: AsyncSession = Depends(get_db)):
    session_id = body.session_id or secrets.token_hex(8)
    temperature = body.temperature if body.temperature is not None else settings.DEFAULT_TEMPERATURE
    top_k       = body.top_k       if body.top_k       is not None else settings.DEFAULT_TOP_K
    max_tokens  = body.max_tokens  if body.max_tokens  is not None else settings.DEFAULT_MAX_TOKENS

    try:
        reply = await inference.generate_reply(
            prompt=body.prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))

    msg = Message(
        session_id=session_id,
        prompt=body.prompt,
        reply=reply,
        temperature=temperature,
        top_k=top_k,
        max_tokens=max_tokens,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    return ApiResponse(data=ChatSendResponse(message=_to_public(msg, feedback=[])))


@router.get("/history", response_model=ApiResponse[list[MessagePublic]])
async def history(
    session_id: Optional[str] = Query(None, max_length=64),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy.orm import selectinload

    stmt = select(Message).options(selectinload(Message.feedback))
    if session_id:
        stmt = stmt.where(Message.session_id == session_id)
    stmt = stmt.order_by(Message.created_at.desc()).limit(limit)

    res = await db.execute(stmt)
    rows = list(res.scalars().unique().all())
    rows.reverse()  # chronological ascending
    return ApiResponse(data=[_to_public(m, feedback=list(m.feedback)) for m in rows])
