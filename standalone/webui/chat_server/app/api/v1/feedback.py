from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_admin
from app.db import get_db
from app.models import Feedback, Message, User
from app.schemas import (ApiResponse, FeedbackCreateRequest, FeedbackPublic,
                         FeedbackUpdateRequest)

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=ApiResponse[FeedbackPublic])
async def create(
    body: FeedbackCreateRequest,
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    msg = await db.get(Message, body.message_id)
    if msg is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Message not found")

    fb = Feedback(
        message_id=body.message_id,
        rating=body.rating,
        corrected_text=body.corrected_text,
        author_id=admin.id,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)

    return ApiResponse(data=FeedbackPublic(
        id=fb.id, rating=fb.rating, corrected_text=fb.corrected_text,
        applied_at=fb.applied_at, created_at=fb.created_at,
    ))


@router.patch("/{feedback_id}", response_model=ApiResponse[FeedbackPublic])
async def update(
    feedback_id: int,
    body: FeedbackUpdateRequest,
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    fb = await db.get(Feedback, feedback_id)
    if fb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback not found")

    if body.rating is not None:
        fb.rating = body.rating
    if body.corrected_text is not None:
        fb.corrected_text = body.corrected_text

    await db.commit()
    await db.refresh(fb)

    return ApiResponse(data=FeedbackPublic(
        id=fb.id, rating=fb.rating, corrected_text=fb.corrected_text,
        applied_at=fb.applied_at, created_at=fb.created_at,
    ))


@router.delete("/{feedback_id}", response_model=ApiResponse[dict])
async def delete(
    feedback_id: int,
    admin: User = Depends(current_admin),
    db: AsyncSession = Depends(get_db),
):
    fb = await db.get(Feedback, feedback_id)
    if fb is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Feedback not found")
    await db.delete(fb)
    await db.commit()
    return ApiResponse(data={"id": feedback_id, "deleted": True})
