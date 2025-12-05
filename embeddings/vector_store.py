"""
Vector Store for Constitutional Agent Semantic Memory

SQLite-based vector store with support for:
- Storing embeddings with metadata
- Similarity search using cosine similarity
- Filtering by agent/country
- Batch operations
"""

import sqlite3
import json
from typing import List, Optional, Tuple
from dataclasses import dataclass
import os

from .ollama_embedder import OllamaEmbedder
from .chunker import Chunk


@dataclass
class SearchResult:
    """A search result with similarity score."""
    chunk_id: str
    content: str
    content_type: str
    section_title: Optional[str]
    section_number: Optional[str]
    similarity: float
    metadata: dict


class VectorStore:
    """
    SQLite-based vector store for constitutional embeddings.

    Provides efficient storage and retrieval of embedded constitutional chunks.
    Uses brute-force cosine similarity search (sufficient for ~1000s of chunks).
    """

    def __init__(self, db_path: str, embedder: Optional[OllamaEmbedder] = None):
        """
        Initialize the vector store.

        Args:
            db_path: Path to the memory database
            embedder: OllamaEmbedder instance (created if not provided)
        """
        self.db_path = db_path
        self.embedder = embedder or OllamaEmbedder()
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the semantic_memory table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT NOT NULL,
                source_document TEXT,
                section_title TEXT,
                section_number TEXT,
                start_position INTEGER,
                end_position INTEGER,
                embedding BLOB NOT NULL,
                embedding_model TEXT NOT NULL,
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                UNIQUE(agent_id, chunk_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memory(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memory(agent_id, content_type)")

        conn.commit()
        conn.close()

    def add_chunk(self, agent_id: str, chunk: Chunk, embedding: Optional[List[float]] = None):
        """
        Add a chunk to the vector store.

        Args:
            agent_id: The agent/country identifier
            chunk: The Chunk object to store
            embedding: Pre-computed embedding (computed if not provided)
        """
        if embedding is None:
            embedding = self.embedder.embed(chunk.content)

        embedding_bytes = self.embedder.embedding_to_bytes(embedding)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO semantic_memory (
                agent_id, chunk_id, content, content_type, source_document,
                section_title, section_number, start_position, end_position,
                embedding, embedding_model, token_count, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            agent_id,
            chunk.chunk_id,
            chunk.content,
            chunk.content_type,
            chunk.source_document,
            chunk.section_title,
            chunk.section_number,
            chunk.start_position,
            chunk.end_position,
            embedding_bytes,
            self.embedder.model_name,
            chunk.token_estimate,
            json.dumps(chunk.metadata)
        ))

        conn.commit()
        conn.close()

    def add_chunks_batch(
        self,
        agent_id: str,
        chunks: List[Chunk],
        show_progress: bool = True
    ) -> int:
        """
        Add multiple chunks to the vector store.

        Args:
            agent_id: The agent/country identifier
            chunks: List of Chunk objects
            show_progress: Whether to print progress

        Returns:
            Number of chunks added
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        count = 0
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            try:
                embedding = self.embedder.embed(chunk.content)
                embedding_bytes = self.embedder.embedding_to_bytes(embedding)

                cursor.execute("""
                    INSERT OR REPLACE INTO semantic_memory (
                        agent_id, chunk_id, content, content_type, source_document,
                        section_title, section_number, start_position, end_position,
                        embedding, embedding_model, token_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    agent_id,
                    chunk.chunk_id,
                    chunk.content,
                    chunk.content_type,
                    chunk.source_document,
                    chunk.section_title,
                    chunk.section_number,
                    chunk.start_position,
                    chunk.end_position,
                    embedding_bytes,
                    self.embedder.model_name,
                    chunk.token_estimate,
                    json.dumps(chunk.metadata)
                ))

                count += 1

                if show_progress and (i + 1) % 10 == 0:
                    print(f"  Embedded {i + 1}/{total} chunks...")

            except Exception as e:
                print(f"  Error embedding chunk {chunk.chunk_id}: {e}")

        conn.commit()
        conn.close()

        if show_progress:
            print(f"  Completed: {count}/{total} chunks embedded")

        return count

    def search(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5,
        content_types: Optional[List[str]] = None,
        min_similarity: float = 0.0
    ) -> List[SearchResult]:
        """
        Search for similar chunks using semantic similarity.

        Args:
            agent_id: The agent/country identifier
            query: The search query text
            top_k: Number of results to return
            content_types: Filter by content types (e.g., ['article', 'preamble'])
            min_similarity: Minimum similarity threshold

        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Get all chunks for this agent
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if content_types:
            placeholders = ','.join('?' * len(content_types))
            cursor.execute(f"""
                SELECT chunk_id, content, content_type, section_title,
                       section_number, embedding, metadata
                FROM semantic_memory
                WHERE agent_id = ? AND content_type IN ({placeholders})
            """, [agent_id] + content_types)
        else:
            cursor.execute("""
                SELECT chunk_id, content, content_type, section_title,
                       section_number, embedding, metadata
                FROM semantic_memory
                WHERE agent_id = ?
            """, (agent_id,))

        results = []
        for row in cursor.fetchall():
            chunk_embedding = self.embedder.bytes_to_embedding(row['embedding'])
            similarity = self.embedder.cosine_similarity(query_embedding, chunk_embedding)

            if similarity >= min_similarity:
                results.append(SearchResult(
                    chunk_id=row['chunk_id'],
                    content=row['content'],
                    content_type=row['content_type'],
                    section_title=row['section_title'],
                    section_number=row['section_number'],
                    similarity=similarity,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))

        conn.close()

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def get_all_chunks(self, agent_id: str) -> List[dict]:
        """Get all chunks for an agent (without embeddings)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT chunk_id, content, content_type, section_title,
                   section_number, token_count, metadata
            FROM semantic_memory
            WHERE agent_id = ?
            ORDER BY id
        """, (agent_id,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return results

    def get_chunk_count(self, agent_id: str) -> int:
        """Get the number of chunks for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM semantic_memory WHERE agent_id = ?",
            (agent_id,)
        )
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def delete_agent_chunks(self, agent_id: str):
        """Delete all chunks for an agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM semantic_memory WHERE agent_id = ?",
            (agent_id,)
        )

        conn.commit()
        conn.close()

    def get_statistics(self) -> dict:
        """Get vector store statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(DISTINCT agent_id) FROM semantic_memory")
        stats['total_agents'] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM semantic_memory")
        stats['total_chunks'] = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(token_count) FROM semantic_memory")
        stats['total_tokens'] = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT agent_id, COUNT(*) as chunk_count, SUM(token_count) as tokens
            FROM semantic_memory
            GROUP BY agent_id
            ORDER BY chunk_count DESC
        """)
        stats['by_agent'] = [dict(row) for row in cursor.fetchall()]

        conn.row_factory = sqlite3.Row
        cursor.execute("""
            SELECT content_type, COUNT(*) as count
            FROM semantic_memory
            GROUP BY content_type
        """)
        stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()
        return stats


def embed_all_constitutions(
    constitutions_db: str,
    memory_db: str,
    show_progress: bool = True
) -> dict:
    """
    Embed all constitutions from the database.

    Args:
        constitutions_db: Path to the constitutions database
        memory_db: Path to the memory database
        show_progress: Whether to print progress

    Returns:
        Statistics about the embedding process
    """
    from .chunker import chunk_all_constitutions

    # Initialize vector store
    store = VectorStore(memory_db)

    stats = {
        'countries': 0,
        'chunks': 0,
        'errors': []
    }

    current_country = None
    current_chunks = []

    for country, chunk in chunk_all_constitutions(constitutions_db):
        if current_country != country:
            # Save previous country's chunks
            if current_country and current_chunks:
                if show_progress:
                    print(f"\nEmbedding {current_country} ({len(current_chunks)} chunks)...")
                try:
                    count = store.add_chunks_batch(current_country, current_chunks, show_progress)
                    stats['chunks'] += count
                    stats['countries'] += 1
                except Exception as e:
                    stats['errors'].append(f"{current_country}: {e}")
                    print(f"  Error: {e}")

            current_country = country
            current_chunks = []

        current_chunks.append(chunk)

    # Don't forget the last country
    if current_country and current_chunks:
        if show_progress:
            print(f"\nEmbedding {current_country} ({len(current_chunks)} chunks)...")
        try:
            count = store.add_chunks_batch(current_country, current_chunks, show_progress)
            stats['chunks'] += count
            stats['countries'] += 1
        except Exception as e:
            stats['errors'].append(f"{current_country}: {e}")

    return stats


if __name__ == "__main__":
    import sys

    # Test the vector store
    print("Testing VectorStore...")

    # Create test database
    test_db = "/tmp/test_vectors.db"
    store = VectorStore(test_db)

    # Create test chunks
    test_chunks = [
        Chunk(
            chunk_id="test_001",
            content="We the People of the United States, in Order to form a more perfect Union",
            content_type="preamble",
            source_document="test",
            section_title="Preamble",
            token_estimate=15
        ),
        Chunk(
            chunk_id="test_002",
            content="All legislative Powers herein granted shall be vested in a Congress",
            content_type="article",
            source_document="test",
            section_title="Article I",
            section_number="1",
            token_estimate=12
        ),
        Chunk(
            chunk_id="test_003",
            content="The executive Power shall be vested in a President of the United States",
            content_type="article",
            source_document="test",
            section_title="Article II",
            section_number="2",
            token_estimate=14
        ),
    ]

    print("\nAdding test chunks...")
    count = store.add_chunks_batch("test_agent", test_chunks)
    print(f"Added {count} chunks")

    print("\nSearching for 'legislative power'...")
    results = store.search("test_agent", "legislative power", top_k=3)
    for r in results:
        print(f"  [{r.similarity:.3f}] {r.section_title}: {r.content[:50]}...")

    print("\nStatistics:")
    stats = store.get_statistics()
    print(f"  Agents: {stats['total_agents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Tokens: {stats['total_tokens']}")

    # Cleanup
    os.remove(test_db)
    print("\nTest complete!")
