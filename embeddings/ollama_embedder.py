"""
Ollama Embedding Integration for Constitutional Agents

Provides local embedding generation using Ollama models.
Supports multiple embedding models with automatic fallback.

Recommended models (in order of preference):
1. nomic-embed-text - Best quality, 768 dimensions
2. mxbai-embed-large - Good quality, 1024 dimensions
3. all-minilm - Fast, lightweight, 384 dimensions
"""

import json
import urllib.request
import urllib.error
from typing import List, Optional, Union
from dataclasses import dataclass
import struct


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    batch_size: int = 32


class OllamaEmbedder:
    """
    Local embedding generator using Ollama.

    Usage:
        embedder = OllamaEmbedder()

        # Single text
        embedding = embedder.embed("The constitution guarantees freedom of speech")

        # Batch embedding
        embeddings = embedder.embed_batch([
            "Article 1: Legislative powers",
            "Article 2: Executive powers",
            "Article 3: Judicial powers"
        ])
    """

    # Known embedding dimensions for common models
    MODEL_DIMENSIONS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
        "bge-large": 1024,
        "bge-base": 768,
    }

    # Fallback model order
    FALLBACK_MODELS = [
        "nomic-embed-text",
        "mxbai-embed-large",
        "all-minilm",
        "snowflake-arctic-embed",
    ]

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedder with optional configuration."""
        self.config = config or EmbeddingConfig()
        self._dimensions: Optional[int] = None
        self._active_model: Optional[str] = None

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions for the active model."""
        if self._dimensions is None:
            # Try to get dimensions from known models
            if self._active_model and self._active_model in self.MODEL_DIMENSIONS:
                self._dimensions = self.MODEL_DIMENSIONS[self._active_model]
            else:
                # Generate a test embedding to determine dimensions
                test_embedding = self.embed("test")
                self._dimensions = len(test_embedding)
        return self._dimensions

    @property
    def model_name(self) -> str:
        """Get the active model name."""
        return self._active_model or self.config.model

    def _make_request(self, endpoint: str, data: dict) -> dict:
        """Make a request to the Ollama API."""
        url = f"{self.config.base_url}{endpoint}"

        request = urllib.request.Request(
            url,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Ollama at {self.config.base_url}: {e}")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8') if e.fp else str(e)
            raise RuntimeError(f"Ollama API error ({e.code}): {error_body}")

    def _check_model_available(self, model: str) -> bool:
        """Check if a model is available in Ollama."""
        try:
            url = f"{self.config.base_url}/api/tags"
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                models = [m.get('name', '').split(':')[0] for m in data.get('models', [])]
                return model in models or f"{model}:latest" in [m.get('name', '') for m in data.get('models', [])]
        except Exception:
            return False

    def _find_available_model(self) -> str:
        """Find an available embedding model, with fallback."""
        # First try the configured model
        if self._check_model_available(self.config.model):
            return self.config.model

        # Try fallback models
        for model in self.FALLBACK_MODELS:
            if self._check_model_available(model):
                print(f"Using fallback model: {model}")
                return model

        # If no model found, return configured and let it fail with helpful error
        return self.config.model

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if self._active_model is None:
            self._active_model = self._find_available_model()

        # Clean and truncate text if needed
        text = text.strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        # Ollama embedding endpoint
        response = self._make_request("/api/embeddings", {
            "model": self._active_model,
            "prompt": text
        })

        embedding = response.get("embedding")
        if embedding is None:
            raise RuntimeError(f"No embedding returned from Ollama: {response}")

        return embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = [self.embed(text) for text in batch]
            embeddings.extend(batch_embeddings)

        return embeddings

    @staticmethod
    def embedding_to_bytes(embedding: List[float]) -> bytes:
        """Convert embedding list to bytes for database storage."""
        return struct.pack(f'{len(embedding)}f', *embedding)

    @staticmethod
    def bytes_to_embedding(data: bytes) -> List[float]:
        """Convert bytes back to embedding list."""
        count = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f'{count}f', data))

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions don't match: {len(vec1)} vs {len(vec2)}")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def find_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find the most similar embeddings to a query.

        Args:
            query_embedding: The query vector
            candidate_embeddings: List of candidate vectors
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.cosine_similarity(query_embedding, candidate)
            similarities.append((i, similarity))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]


def check_ollama_status(base_url: str = "http://localhost:11434") -> dict:
    """
    Check Ollama server status and available models.

    Returns:
        Dict with status information
    """
    status = {
        "online": False,
        "models": [],
        "embedding_models": [],
        "error": None
    }

    try:
        url = f"{base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            status["online"] = True
            status["models"] = [m.get('name', '') for m in data.get('models', [])]

            # Filter for known embedding models
            embedding_models = []
            for model in status["models"]:
                model_base = model.split(':')[0]
                if model_base in OllamaEmbedder.MODEL_DIMENSIONS:
                    embedding_models.append(model)
            status["embedding_models"] = embedding_models

    except urllib.error.URLError as e:
        status["error"] = f"Cannot connect to Ollama: {e}"
    except Exception as e:
        status["error"] = str(e)

    return status


if __name__ == "__main__":
    # Test the embedder
    print("Checking Ollama status...")
    status = check_ollama_status()
    print(f"Online: {status['online']}")
    print(f"Models: {status['models']}")
    print(f"Embedding models: {status['embedding_models']}")

    if status['online'] and status['embedding_models']:
        print("\nTesting embedder...")
        embedder = OllamaEmbedder()

        test_texts = [
            "We the People of the United States",
            "All legislative Powers herein granted shall be vested in a Congress",
            "The executive Power shall be vested in a President"
        ]

        for text in test_texts:
            embedding = embedder.embed(text)
            print(f"Text: {text[:50]}...")
            print(f"Embedding dimensions: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
            print()
    else:
        print(f"\nCannot test: {status.get('error', 'No embedding models available')}")
        print("\nTo install an embedding model, run:")
        print("  ollama pull nomic-embed-text")
