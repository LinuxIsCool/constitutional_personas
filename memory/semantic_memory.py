"""
Semantic Memory for Constitutional Agents

Stores factual knowledge and constitutional text with embeddings for
RAG (Retrieval Augmented Generation) retrieval. This is the primary
memory for constitutional knowledge.

The semantic memory integrates with the vector store to provide:
- Embedded constitutional chunks
- Similarity-based retrieval
- Contextual knowledge augmentation
"""

import sqlite3
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings.vector_store import VectorStore, SearchResult
from embeddings.ollama_embedder import OllamaEmbedder
from embeddings.chunker import ConstitutionChunker, Chunk


@dataclass
class SemanticEntry:
    """A semantic memory entry."""
    chunk_id: str
    content: str
    content_type: str
    section_title: Optional[str]
    section_number: Optional[str]
    token_count: int
    metadata: Dict[str, Any]


class SemanticMemory:
    """
    Semantic memory for constitutional knowledge with RAG retrieval.

    This memory type is designed for:
    - Storing embedded constitutional text chunks
    - Similarity-based knowledge retrieval
    - Augmenting responses with relevant constitutional passages
    """

    def __init__(
        self,
        db_path: str,
        agent_id: str,
        embedder: Optional[OllamaEmbedder] = None
    ):
        """
        Initialize semantic memory.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent (usually country name)
            embedder: OllamaEmbedder instance (created if not provided)
        """
        self.db_path = db_path
        self.agent_id = agent_id
        self.embedder = embedder
        self._vector_store: Optional[VectorStore] = None

    @property
    def vector_store(self) -> VectorStore:
        """Lazy initialization of vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(self.db_path, self.embedder)
        return self._vector_store

    def is_initialized(self) -> bool:
        """Check if semantic memory has been initialized with chunks."""
        return self.vector_store.get_chunk_count(self.agent_id) > 0

    def initialize_from_constitution(
        self,
        full_text: str,
        source_url: str = "",
        show_progress: bool = True
    ) -> int:
        """
        Initialize semantic memory from constitution text.

        Args:
            full_text: The full constitutional text
            source_url: Source URL for the constitution
            show_progress: Whether to show progress

        Returns:
            Number of chunks created
        """
        # Clear existing chunks
        self.vector_store.delete_agent_chunks(self.agent_id)

        # Create chunker
        chunker = ConstitutionChunker(
            max_chunk_size=500,
            min_chunk_size=100,
            overlap_size=50,
            strategy="hybrid"
        )

        # Generate chunks
        chunks = list(chunker.chunk_constitution(full_text, self.agent_id, source_url))

        if show_progress:
            print(f"Generated {len(chunks)} chunks for {self.agent_id}")

        # Add chunks to vector store
        count = self.vector_store.add_chunks_batch(self.agent_id, chunks, show_progress)

        return count

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        content_types: Optional[List[str]] = None,
        min_similarity: float = 0.3
    ) -> List[SearchResult]:
        """
        Retrieve relevant constitutional passages for a query.

        Args:
            query: The search query
            top_k: Number of results to return
            content_types: Filter by content types
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects
        """
        return self.vector_store.search(
            agent_id=self.agent_id,
            query=query,
            top_k=top_k,
            content_types=content_types,
            min_similarity=min_similarity
        )

    def retrieve_by_topic(self, topic: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve passages related to a specific topic.

        Convenience method with reasonable defaults.
        """
        return self.retrieve(query=topic, top_k=top_k, min_similarity=0.25)

    def retrieve_articles(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Retrieve only article-type content."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            content_types=['article'],
            min_similarity=0.25
        )

    def retrieve_preamble(self) -> List[SearchResult]:
        """Retrieve preamble content."""
        return self.vector_store.search(
            agent_id=self.agent_id,
            query="preamble founding principles values",
            top_k=3,
            content_types=['preamble'],
            min_similarity=0.0  # Get all preamble chunks
        )

    def add_knowledge(
        self,
        content: str,
        content_type: str = "fact",
        section_title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add additional knowledge to semantic memory.

        Args:
            content: The knowledge content
            content_type: Type of content
            section_title: Optional title
            metadata: Additional metadata
        """
        chunk = Chunk(
            chunk_id=f"{self.agent_id}_knowledge_{hash(content) % 10000:04d}",
            content=content,
            content_type=content_type,
            source_document="added_knowledge",
            section_title=section_title,
            token_estimate=len(content) // 4,
            metadata=metadata or {}
        )

        self.vector_store.add_chunk(self.agent_id, chunk)

    def get_all_chunks(self) -> List[SemanticEntry]:
        """Get all semantic memory chunks (without embeddings)."""
        chunks = self.vector_store.get_all_chunks(self.agent_id)
        return [
            SemanticEntry(
                chunk_id=c['chunk_id'],
                content=c['content'],
                content_type=c['content_type'],
                section_title=c['section_title'],
                section_number=c['section_number'],
                token_count=c['token_count'] or 0,
                metadata=json.loads(c['metadata']) if c['metadata'] else {}
            )
            for c in chunks
        ]

    def count(self) -> int:
        """Get the number of chunks in semantic memory."""
        return self.vector_store.get_chunk_count(self.agent_id)

    def clear(self):
        """Clear all semantic memory for this agent."""
        self.vector_store.delete_agent_chunks(self.agent_id)

    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get relevant context for a query, formatted for prompt injection.

        Args:
            query: The user's query
            max_tokens: Maximum tokens to return

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, top_k=10, min_similarity=0.25)

        if not results:
            return "No relevant constitutional passages found."

        context_parts = []
        total_tokens = 0

        for result in results:
            # Estimate tokens
            chunk_tokens = len(result.content) // 4

            if total_tokens + chunk_tokens > max_tokens:
                break

            section = result.section_title or result.content_type.title()
            context_parts.append(f"[{section}]\n{result.content}")
            total_tokens += chunk_tokens

        return "\n\n".join(context_parts)

    def get_summary(self) -> str:
        """Get a summary of semantic memory for context injection."""
        chunk_count = self.count()

        if chunk_count == 0:
            return "Semantic memory not initialized."

        # Get content type distribution
        chunks = self.get_all_chunks()
        type_counts: Dict[str, int] = {}
        for chunk in chunks:
            type_counts[chunk.content_type] = type_counts.get(chunk.content_type, 0) + 1

        type_str = ", ".join(f"{k}: {v}" for k, v in sorted(type_counts.items()))

        return f"Constitutional knowledge: {chunk_count} embedded passages ({type_str})"


def initialize_all_semantic_memories(
    constitutions_db: str,
    memory_db: str,
    show_progress: bool = True
) -> Dict[str, int]:
    """
    Initialize semantic memory for all constitutions.

    Args:
        constitutions_db: Path to the constitutions database
        memory_db: Path to the memory database
        show_progress: Whether to show progress

    Returns:
        Dict mapping country to chunk count
    """
    conn = sqlite3.connect(constitutions_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT country, full_text, source_url FROM constitutions")

    results = {}
    embedder = OllamaEmbedder()  # Share embedder across all

    for row in cursor.fetchall():
        country = row['country']
        if show_progress:
            print(f"\n{'='*60}")
            print(f"Initializing semantic memory for: {country}")
            print(f"{'='*60}")

        memory = SemanticMemory(memory_db, country, embedder)
        count = memory.initialize_from_constitution(
            row['full_text'],
            row['source_url'] or '',
            show_progress
        )
        results[country] = count

    conn.close()

    if show_progress:
        print(f"\n{'='*60}")
        print("INITIALIZATION COMPLETE")
        print(f"{'='*60}")
        total = sum(results.values())
        print(f"Total countries: {len(results)}")
        print(f"Total chunks: {total}")

    return results
