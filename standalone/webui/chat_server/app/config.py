"""
Application settings loaded from .env via pydantic-settings.
"""
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVER_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(SERVER_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    PORT: int = 8001
    ENVIRONMENT: str = "development"
    CORS_ORIGINS_RAW: str = Field("http://localhost:5501", alias="CORS_ORIGINS")

    MODEL_PATH: str = "../../model.pt"
    TOKENIZER_PATH: str = "../../tokenizer.json"
    STANDALONE_DIR: str = "../.."

    DATABASE_URL: str = "sqlite+aiosqlite:///./data/chat.db"

    JWT_SECRET: str = "change-me"
    JWT_ACCESS_TOKEN_EXPIRE_HOURS: int = 24
    JWT_ALGORITHM: str = "HS256"

    ADMIN_EMAIL: str = "admin@physisml.local"
    ADMIN_PASSWORD: str = "admin123"

    DEFAULT_MAX_TOKENS: int = 80
    DEFAULT_TEMPERATURE: float = 0.8
    DEFAULT_TOP_K: int = 40

    TRAIN_LR: float = 1e-5
    TRAIN_STEPS_PER_SAMPLE: int = 3

    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]

    def resolve(self, relative: str) -> Path:
        """Resolve a path relative to chat_server/."""
        p = Path(relative)
        return p if p.is_absolute() else (SERVER_ROOT / p).resolve()


settings = Settings()
