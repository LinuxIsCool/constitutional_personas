"""
Persistent Memory for Constitutional Agents

Long-term storage of facts, beliefs, preferences, and learned information.
This memory survives across all sessions and represents the agent's
core knowledge and identity.

Examples:
- Beliefs: "Individual liberty is paramount"
- Facts: "I was established in 1787"
- Preferences: "I prefer strict constitutional interpretation"
- Learned: "Users often ask about the Bill of Rights"
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PersistentMemoryEntry:
    """A persistent memory entry."""
    id: int
    key: str
    value: str
    memory_type: str  # fact, belief, preference, learned
    importance: float
    confidence: float
    source: Optional[str]
    created_at: datetime
    updated_at: datetime
    access_count: int
    metadata: Dict[str, Any]


class PersistentMemory:
    """
    Persistent memory store for long-term information.

    This memory type is designed for:
    - Core identity and beliefs
    - Learned facts and preferences
    - Information that should persist across sessions
    - High-value knowledge with confidence scores
    """

    def __init__(self, db_path: str, agent_id: str):
        """
        Initialize persistent memory.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent
        """
        self.db_path = db_path
        self.agent_id = agent_id
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the persistent_memory table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persistent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                memory_type TEXT DEFAULT 'fact',
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 0.8,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP,
                metadata TEXT,
                UNIQUE(agent_id, memory_key)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persistent_agent ON persistent_memory(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_persistent_key ON persistent_memory(agent_id, memory_key)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def store(
        self,
        key: str,
        value: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        confidence: float = 0.8,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Store a persistent memory.

        Args:
            key: Unique key for this memory
            value: The memory content
            memory_type: Type of memory (fact, belief, preference, learned)
            importance: Importance score 0.0-1.0
            confidence: Confidence score 0.0-1.0
            source: Where this memory came from
            metadata: Additional metadata

        Returns:
            The memory ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO persistent_memory (
                agent_id, memory_key, memory_value, memory_type,
                importance, confidence, source, metadata, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, memory_key) DO UPDATE SET
                memory_value = excluded.memory_value,
                memory_type = excluded.memory_type,
                importance = excluded.importance,
                confidence = excluded.confidence,
                source = excluded.source,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at
        """, (
            self.agent_id, key, value, memory_type,
            importance, confidence, source,
            json.dumps(metadata) if metadata else None, now
        ))

        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return memory_id

    def recall(self, key: str) -> Optional[PersistentMemoryEntry]:
        """
        Recall a specific memory by key.

        Also increments the access count.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM persistent_memory
            WHERE agent_id = ? AND memory_key = ?
        """, (self.agent_id, key))

        row = cursor.fetchone()

        if row:
            # Update access count
            cursor.execute("""
                UPDATE persistent_memory
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), row['id']))
            conn.commit()

            entry = PersistentMemoryEntry(
                id=row['id'],
                key=row['memory_key'],
                value=row['memory_value'],
                memory_type=row['memory_type'],
                importance=row['importance'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                access_count=row['access_count'] + 1,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
            conn.close()
            return entry

        conn.close()
        return None

    def recall_by_type(self, memory_type: str, limit: int = 10) -> List[PersistentMemoryEntry]:
        """Recall memories of a specific type, sorted by importance."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM persistent_memory
            WHERE agent_id = ? AND memory_type = ?
            ORDER BY importance DESC, confidence DESC
            LIMIT ?
        """, (self.agent_id, memory_type, limit))

        entries = []
        for row in cursor.fetchall():
            entries.append(PersistentMemoryEntry(
                id=row['id'],
                key=row['memory_key'],
                value=row['memory_value'],
                memory_type=row['memory_type'],
                importance=row['importance'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                access_count=row['access_count'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))

        conn.close()
        return entries

    def recall_important(self, min_importance: float = 0.7, limit: int = 20) -> List[PersistentMemoryEntry]:
        """Recall the most important memories."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM persistent_memory
            WHERE agent_id = ? AND importance >= ?
            ORDER BY importance DESC, confidence DESC
            LIMIT ?
        """, (self.agent_id, min_importance, limit))

        entries = []
        for row in cursor.fetchall():
            entries.append(PersistentMemoryEntry(
                id=row['id'],
                key=row['memory_key'],
                value=row['memory_value'],
                memory_type=row['memory_type'],
                importance=row['importance'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                access_count=row['access_count'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))

        conn.close()
        return entries

    def search(self, query: str, limit: int = 10) -> List[PersistentMemoryEntry]:
        """Search memories by content (simple LIKE search)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM persistent_memory
            WHERE agent_id = ? AND (
                memory_key LIKE ? OR memory_value LIKE ?
            )
            ORDER BY importance DESC
            LIMIT ?
        """, (self.agent_id, f'%{query}%', f'%{query}%', limit))

        entries = []
        for row in cursor.fetchall():
            entries.append(PersistentMemoryEntry(
                id=row['id'],
                key=row['memory_key'],
                value=row['memory_value'],
                memory_type=row['memory_type'],
                importance=row['importance'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                access_count=row['access_count'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))

        conn.close()
        return entries

    def update_importance(self, key: str, importance: float):
        """Update the importance of a memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE persistent_memory
            SET importance = ?, updated_at = ?
            WHERE agent_id = ? AND memory_key = ?
        """, (importance, datetime.now().isoformat(), self.agent_id, key))

        conn.commit()
        conn.close()

    def update_confidence(self, key: str, confidence: float):
        """Update the confidence of a memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE persistent_memory
            SET confidence = ?, updated_at = ?
            WHERE agent_id = ? AND memory_key = ?
        """, (confidence, datetime.now().isoformat(), self.agent_id, key))

        conn.commit()
        conn.close()

    def forget(self, key: str):
        """Remove a memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM persistent_memory
            WHERE agent_id = ? AND memory_key = ?
        """, (self.agent_id, key))

        conn.commit()
        conn.close()

    def get_all(self) -> List[PersistentMemoryEntry]:
        """Get all persistent memories for this agent."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM persistent_memory
            WHERE agent_id = ?
            ORDER BY importance DESC
        """, (self.agent_id,))

        entries = []
        for row in cursor.fetchall():
            entries.append(PersistentMemoryEntry(
                id=row['id'],
                key=row['memory_key'],
                value=row['memory_value'],
                memory_type=row['memory_type'],
                importance=row['importance'],
                confidence=row['confidence'],
                source=row['source'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                access_count=row['access_count'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))

        conn.close()
        return entries

    def count(self) -> int:
        """Get the number of persistent memories."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM persistent_memory WHERE agent_id = ?
        """, (self.agent_id,))

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_summary(self) -> str:
        """Get a summary of persistent memory for context injection."""
        beliefs = self.recall_by_type('belief', limit=5)
        facts = self.recall_by_type('fact', limit=5)
        preferences = self.recall_by_type('preference', limit=3)

        summary_parts = []

        if beliefs:
            summary_parts.append("Core Beliefs:")
            for b in beliefs:
                summary_parts.append(f"  - {b.value}")

        if facts:
            summary_parts.append("Key Facts:")
            for f in facts:
                summary_parts.append(f"  - {f.value}")

        if preferences:
            summary_parts.append("Preferences:")
            for p in preferences:
                summary_parts.append(f"  - {p.value}")

        return '\n'.join(summary_parts) if summary_parts else "No persistent memories stored."
