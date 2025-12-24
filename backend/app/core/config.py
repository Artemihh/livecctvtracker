from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embedding_retention_seconds: int = 10

settings = Settings()
