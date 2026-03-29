"""환경변수 및 설정 관리."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # LLM Provider 선택: anthropic | gemini
    llm_provider: str = "anthropic"

    # Anthropic
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # SQLite (Milvus 대체)
    sqlite_path: str = "data/knowledge.db"

    # Agent
    max_steps: int = 5
    vector_search_top_k: int = 5
    web_search_top_k: int = 5

    # 인증
    admin_username: str = "admin"
    admin_password: str = "changeme"
    auth_token: str = "changeme-token"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()

