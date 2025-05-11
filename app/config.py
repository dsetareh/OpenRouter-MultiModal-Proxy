from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENROUTER_API_KEY: str
    DEFAULT_TEXT_MODEL: str = "openai/gpt-3.5-turbo"
    VISION_MODEL: str = "mistralai/mistral-small-3.1-24b-instruct"
    LOG_FILE_PATH: str = "router.log.json"
    OPENROUTER_REFERER: str | None = None
    OPENROUTER_X_TITLE: str | None = None
    DATABASE_URL: str = "sqlite+aiosqlite:///./openrouter-multimodal-proxy.db"
    WHISPER_MODEL_NAME: str = "distil-medium.en"
    WHISPER_DEVICE: str = "cpu"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
