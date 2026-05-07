from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    jwt_secret: str
    jwt_expires_minutes: int = 60 * 24 * 7
    # Set True behind HTTPS in production so browsers send the auth cookie only over TLS.
    auth_cookie_secure: bool = False
    # Persist rate-limit counters in DB (survives process restarts; use with single-node SQLite/Postgres).
    rate_limit_persist: bool = True
    # Optional: ask the LLM to refine CV role suggestions after keyword baseline (uses OpenAI bill).
    cv_llm_enrich: bool = False
    # One structured LLM pass: role/sector refinement + short evaluator (strengths/weaknesses/fit). Saves tokens vs enrich+screening separately.
    cv_llm_screening: bool = True
    # Optional override for CV screening only (empty = same as llm_model).
    cv_screening_model: str = ""

    database_url: str

    openai_api_key: str

    llm_model: str = "gpt-4o-mini"
    transcribe_model: str = "gpt-4o-mini-transcribe"
    tts_model: str = "gpt-4o-mini-tts"
    embedding_model: str = "text-embedding-3-small"

    chroma_dir: str = "./chroma"
    cors_origins: str = ""
    # Optional: redis://... for distributed rate limits across app instances.
    redis_url: str = ""
    # Fuse OpenAI embeddings with keyword scores for CV role suggestions.
    cv_use_embedding: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.cors_origins or "").strip()
        if not raw:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]


settings = Settings()