"""OpenAI embedding client with batching and exponential backoff retry."""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional
import warnings

from openai import OpenAI
from openai import (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)

from codesem.config.settings import get_settings


class OpenAIEmbedder:
    """
    OpenAI embedding client wrapper.

    Responsibilities:
    - Batch embedding requests
    - Optional dimension override
    - Basic exponential backoff retry
    - Deterministic ordering of outputs

    This class does NOT cache results.
    Caching should happen at a higher layer (e.g., via content_hash in DB).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 50,
        max_retries: int = 5,
        backoff_base: float = 0.5,
    ) -> None:
        settings = get_settings()

        self.api_key = settings.openai_api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model or settings.embedding_model
        self.dimensions = dimensions if dimensions is not None else settings.embedding_dimensions

        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single string.
        """
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """
        Embed multiple strings.

        Returns embeddings in the same order as input.
        """
        text_list = list(texts)
        if not text_list:
            return []

        all_embeddings: List[List[float]] = []

        total = len(text_list)
        num_batches = math.ceil(total / self.batch_size)

        for i in range(num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, total)
            batch = text_list[start:end]

            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        """
        Embed a batch with exponential backoff.
        """
        attempt = 0

        while True:
            try:
                kwargs = {
                    "model": self.model,
                    "input": batch,
                }

                # Only pass dimensions if explicitly set
                if self.dimensions is not None:
                    kwargs["dimensions"] = self.dimensions

                response = self.client.embeddings.create(**kwargs)

                # Preserve order: API returns in same order as input
                return [item.embedding for item in response.data]

            except (RateLimitError, APIConnectionError, APITimeoutError) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        f"Embedding failed after {self.max_retries} retries."
                    ) from e

                sleep_time = self.backoff_base * (2 ** (attempt - 1))
                warnings.warn(
                    f"Embedding batch failed with transient error "
                    f"(attempt {attempt}/{self.max_retries}). "
                    f"Retrying in {sleep_time:.2f}s...",
                    RuntimeWarning,
                )
                time.sleep(sleep_time)

            # Non-transient errors (e.g., auth, invalid request) should not retry
            except Exception:
                raise
