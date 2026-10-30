"""CPU-only OpenAI-compatible embedding service backed by FastEmbed."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastembed import TextEmbedding
from pydantic import BaseModel, Field, field_validator

LOGGER = logging.getLogger("embedding-server")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

MAX_CONCURRENCY_LIMIT = 8


def _positive_int(name: str, default: int, maximum: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be positive, got {value}")
    if maximum is not None:
        value = min(value, maximum)
    return value


@dataclass(frozen=True)
class Settings:
    model_name: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    cache_dir: Path = Path(os.getenv("FASTEMBED_CACHE_DIR", "~/.cache/fastembed")).expanduser()
    threads: int = _positive_int("EMBEDDING_THREADS", max(1, (os.cpu_count() or 1) // 2), 32)
    batch_size: int = _positive_int("EMBEDDING_BATCH_SIZE", 16, 128)
    max_concurrency: int = _positive_int("EMBEDDING_MAX_CONCURRENCY", MAX_CONCURRENCY_LIMIT, MAX_CONCURRENCY_LIMIT)


SETTINGS = Settings()


class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float"] = "float"
    dimensions: int | None = Field(default=None, ge=1)

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value
        if not texts:
            raise ValueError("input must not be empty")
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("input must contain only strings")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info(
        "loading CPU embedding model=%s cache_dir=%s threads=%s batch_size=%s max_concurrency=%s",
        SETTINGS.model_name,
        SETTINGS.cache_dir,
        SETTINGS.threads,
        SETTINGS.batch_size,
        SETTINGS.max_concurrency,
    )
    SETTINGS.cache_dir.mkdir(parents=True, exist_ok=True)
    model = await run_in_threadpool(_load_model)
    app.state.embedding_model = model
    app.state.embedding_semaphore = asyncio.Semaphore(SETTINGS.max_concurrency)
    LOGGER.info("embedding model ready: %s", SETTINGS.model_name)
    try:
        yield
    finally:
        app.state.embedding_model = None


app = FastAPI(title="Local CPU Embedding Service", version="1.0.0", lifespan=lifespan)


def _load_model() -> TextEmbedding:
    model = TextEmbedding(
        model_name=SETTINGS.model_name,
        cache_dir=str(SETTINGS.cache_dir),
        threads=SETTINGS.threads,
        providers=["CPUExecutionProvider"],
        cuda=False,
    )
    # Force ONNX initialization before reporting readiness.
    list(model.embed(["warmup"], batch_size=1))
    return model


def _embed_sync(model: TextEmbedding, texts: list[str]) -> tuple[list[list[float]], int]:
    vectors = list(model.embed(texts, batch_size=SETTINGS.batch_size))
    if len(vectors) != len(texts):
        raise RuntimeError(f"embedding result count mismatch: expected {len(texts)}, got {len(vectors)}")
    token_count = int(model.model.token_count(texts, batch_size=SETTINGS.batch_size))
    return [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors], token_count


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    ready = getattr(request.app.state, "embedding_model", None) is not None
    if not ready:
        raise HTTPException(status_code=503, detail="embedding model is not ready")
    return {
        "status": "ok",
        "model": SETTINGS.model_name,
        "backend": "CPUExecutionProvider",
        "max_concurrency": SETTINGS.max_concurrency,
        "threads": SETTINGS.threads,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": SETTINGS.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "local-fastembed",
            }
        ],
    }


@app.post("/v1/embeddings")
async def create_embeddings(payload: EmbeddingRequest, request: Request) -> dict[str, object]:
    model: TextEmbedding | None = getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="embedding model is not ready")
    if payload.model != SETTINGS.model_name:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported model {payload.model!r}; expected {SETTINGS.model_name!r}",
        )
    if payload.dimensions is not None:
        raise HTTPException(status_code=400, detail="the dimensions parameter is not supported by this model")

    texts = [payload.input] if isinstance(payload.input, str) else list(payload.input)
    semaphore: asyncio.Semaphore = request.app.state.embedding_semaphore
    try:
        async with semaphore:
            vectors, token_count = await run_in_threadpool(_embed_sync, model, texts)
    except Exception as exc:
        LOGGER.exception("embedding request failed")
        raise HTTPException(status_code=500, detail="embedding inference failed") from exc

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": vector,
                "index": index,
            }
            for index, vector in enumerate(vectors)
        ],
        "model": SETTINGS.model_name,
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count,
        },
    }
