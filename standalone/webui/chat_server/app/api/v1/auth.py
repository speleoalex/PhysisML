from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (create_access_token, current_user_optional,
                      verify_password)
from app.db import get_db
from app.models import User
from app.schemas import (ApiResponse, LoginRequest, TokenPayload, UserPublic)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=ApiResponse[TokenPayload])
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(User).where(User.email == body.email))
    user = res.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    token, expires = create_access_token(user.id, user.email)
    return ApiResponse(data=TokenPayload(
        access_token=token,
        expires_in=expires,
        user=UserPublic(id=user.id, email=user.email, is_admin=user.is_admin),
    ))


@router.get("/me", response_model=ApiResponse[UserPublic])
async def me(user: User | None = Depends(current_user_optional)):
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return ApiResponse(data=UserPublic(id=user.id, email=user.email,
                                       is_admin=user.is_admin))
