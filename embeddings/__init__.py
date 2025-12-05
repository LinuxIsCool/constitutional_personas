"""
Embeddings module for Constitutional Agents.

Provides local embedding via Ollama and vector operations for RAG retrieval.
"""

from .ollama_embedder import OllamaEmbedder
from .chunker import ConstitutionChunker
from .vector_store import VectorStore

__all__ = ['OllamaEmbedder', 'ConstitutionChunker', 'VectorStore']
