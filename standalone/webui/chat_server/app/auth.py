"""
Password hashing, JWT access tokens, FastAPI auth dependencies.
"""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models import User


_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(pw: str) -> str:
    return _pwd.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    return _pwd.verify(pw, hashed)


def create_access_token(user_id: int, email: str) -> tuple[str, int]:
    expires_seconds = settings.JWT_ACCESS_TOKEN_EXPIRE_HOURS * 3600
    exp = datetime.utcnow() + timedelta(seconds=expires_seconds)
    payload = {"sub": str(user_id), "email": email, "exp": exp}
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, expires_seconds


def _decode(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None


async def current_user_optional(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    payload = _decode(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    result = await db.execute(select(User).where(User.id == int(user_id)))
    return result.scalar_one_or_none()


async def current_admin(
    user: Optional[User] = Depends(current_user_optional),
) -> User:
    if user is None or not user.is_admin:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Admin login required")
    return user
