from pydantic import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALGORITHM: str = "HS256"
    DATABASE_URL: str
    ALLOWED_ORIGINS: list = ["*"]
    # Add more as needed

    class Config:
        env_file = ".env"

settings = Settings()