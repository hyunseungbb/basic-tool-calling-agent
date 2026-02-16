"""환경변수 및 설정 관리."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정."""

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Milvus
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Agent
    max_steps: int = 5
    vector_search_top_k: int = 5
    web_search_top_k: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

