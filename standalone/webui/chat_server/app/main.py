"""
FastAPI entrypoint: init DB, ensure admin user, load model into memory,
mount routers, CORS.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import inference
from app.api.v1.router import api_router
from app.auth import hash_password
from app.config import settings
from app.db import AsyncSessionLocal, close_db, init_db
from app.models import User


async def ensure_admin() -> None:
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(User).where(User.is_admin == True))  # noqa: E712
        if res.scalar_one_or_none():
            return
        db.add(User(
            email=settings.ADMIN_EMAIL,
            password_hash=hash_password(settings.ADMIN_PASSWORD),
            is_admin=True,
        ))
        await db.commit()
        print(f"Default admin created: {settings.ADMIN_EMAIL}")


def warn_about_defaults() -> None:
    """
    Say out loud when the server is running on the shipped placeholders.

    config.py has to have defaults for JWT_SECRET and ADMIN_PASSWORD or the
    app will not start without a .env, which makes trying it out needlessly
    hard. The failure mode is someone leaving them in place and exposing the
    port: 'change-me' signs every token, so anyone who reads this file can
    mint an admin one. Refusing to start would be wrong for a local demo;
    starting silently is what actually gets people hurt.
    """
    weak = [name for name, placeholder in
            (("JWT_SECRET", "change-me"), ("ADMIN_PASSWORD", "admin123"))
            if getattr(settings, name) == placeholder]
    if not weak:
        return
    print("=" * 68)
    print("  WARNING: still using the shipped default for: " + ", ".join(weak))
    print("  Anyone who has read the source can sign an admin token.")
    print("  Fine on localhost, never on an exposed port. Set them in")
    print("  chat_server/.env (see .env.example) before binding to 0.0.0.0.")
    print("=" * 68)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"PhysisML chat_server starting ({settings.ENVIRONMENT})")
    warn_about_defaults()
    await init_db()
    await ensure_admin()
    inference.load()
    info = inference.model_info()
    print(f"Model loaded: {info['num_params']:,} params, "
          f"vocab {info['active_vocab_size']}/{info['vocab_size']}")
    yield
    await close_db()
    print("chat_server shutdown")


app = FastAPI(
    title="PhysisML Chat API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(StarletteHTTPException)
async def http_exc_handler(_: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "data": None,
            "error": {"code": f"HTTP_{exc.status_code}",
                      "message": str(exc.detail)},
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "data": None,
            "error": {"code": "VALIDATION_ERROR", "message": str(exc.errors())},
        },
    )


@app.exception_handler(Exception)
async def unhandled_exc_handler(_: Request, exc: Exception):
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc) if settings.ENVIRONMENT == "development"
                           else "Internal server error",
            },
        },
    )


app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
