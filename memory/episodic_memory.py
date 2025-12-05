"""
Episodic Memory for Constitutional Agents

Stores timestamped experiences, interactions, and events.
This memory allows agents to "remember when" something happened,
providing temporal context for their responses.

Examples:
- "On Dec 4, 2025, a user asked me about freedom of speech"
- "I previously explained the commerce clause to a student"
- "There was a heated discussion about gun rights last week"
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class Episode:
    """An episodic memory entry."""
    id: int
    episode_type: str  # conversation, query, reflection, event
    summary: str
    content: str
    participants: List[str]
    emotional_valence: float  # -1.0 to 1.0
    importance: float
    timestamp: datetime
    duration_seconds: Optional[int]
    location: Optional[str]  # topic/domain
    tags: List[str]
    linked_episodes: List[int]
    metadata: Dict[str, Any]


class EpisodicMemory:
    """
    Episodic memory store for experiences and events.

    This memory type is designed for:
    - Recording significant interactions
    - Temporal reasoning ("When did we discuss X?")
    - Building rapport through remembered experiences
    - Learning from past successes and failures
    """

    def __init__(self, db_path: str, agent_id: str):
        """
        Initialize episodic memory.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent
        """
        self.db_path = db_path
        self.agent_id = agent_id
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the episodic_memory table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT,
                episode_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                participants TEXT,
                emotional_valence REAL,
                importance REAL DEFAULT 0.5,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds INTEGER,
                location TEXT,
                tags TEXT,
                linked_episodes TEXT,
                embedding BLOB,
                metadata TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_agent ON episodic_memory(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_time ON episodic_memory(agent_id, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic_memory(agent_id, episode_type)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record(
        self,
        episode_type: str,
        summary: str,
        content: str,
        session_id: Optional[str] = None,
        participants: Optional[List[str]] = None,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
        duration_seconds: Optional[int] = None,
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record a new episode.

        Args:
            episode_type: Type of episode (conversation, query, reflection, event)
            summary: Brief summary of the episode
            content: Full content/details
            session_id: Associated session ID
            participants: List of participants
            emotional_valence: Emotional tone -1.0 to 1.0
            importance: Importance score 0.0-1.0
            duration_seconds: How long the episode lasted
            location: Topic/domain context
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            The episode ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO episodic_memory (
                agent_id, session_id, episode_type, summary, content,
                participants, emotional_valence, importance, duration_seconds,
                location, tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.agent_id, session_id, episode_type, summary, content,
            json.dumps(participants) if participants else None,
            emotional_valence, importance, duration_seconds, location,
            json.dumps(tags) if tags else None,
            json.dumps(metadata) if metadata else None
        ))

        episode_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return episode_id

    def record_conversation(
        self,
        summary: str,
        content: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        emotional_valence: float = 0.0,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> int:
        """Convenience method for recording conversation episodes."""
        participants = ['user']
        if user_id:
            participants = [user_id]

        return self.record(
            episode_type='conversation',
            summary=summary,
            content=content,
            session_id=session_id,
            participants=participants,
            emotional_valence=emotional_valence,
            importance=importance,
            tags=tags
        )

    def record_reflection(
        self,
        summary: str,
        content: str,
        importance: float = 0.6,
        tags: Optional[List[str]] = None
    ) -> int:
        """Record a reflection episode (agent thinking about itself)."""
        return self.record(
            episode_type='reflection',
            summary=summary,
            content=content,
            emotional_valence=0.0,
            importance=importance,
            location='self-reflection',
            tags=tags
        )

    def recall_recent(self, limit: int = 10, episode_type: Optional[str] = None) -> List[Episode]:
        """
        Recall recent episodes.

        Args:
            limit: Maximum number of episodes to return
            episode_type: Filter by episode type

        Returns:
            List of recent episodes
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        if episode_type:
            cursor.execute("""
                SELECT * FROM episodic_memory
                WHERE agent_id = ? AND episode_type = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.agent_id, episode_type, limit))
        else:
            cursor.execute("""
                SELECT * FROM episodic_memory
                WHERE agent_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (self.agent_id, limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def recall_by_timeframe(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None
    ) -> List[Episode]:
        """
        Recall episodes within a timeframe.

        Args:
            start_time: Start of the timeframe
            end_time: End of the timeframe (defaults to now)

        Returns:
            List of episodes in the timeframe
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        end_time = end_time or datetime.now()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        """, (self.agent_id, start_time.isoformat(), end_time.isoformat()))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def recall_today(self) -> List[Episode]:
        """Recall today's episodes."""
        start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.recall_by_timeframe(start_of_day)

    def recall_last_n_days(self, days: int) -> List[Episode]:
        """Recall episodes from the last N days."""
        start_time = datetime.now() - timedelta(days=days)
        return self.recall_by_timeframe(start_time)

    def recall_by_location(self, location: str, limit: int = 10) -> List[Episode]:
        """Recall episodes by topic/domain location."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND location LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.agent_id, f'%{location}%', limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def recall_by_tag(self, tag: str, limit: int = 10) -> List[Episode]:
        """Recall episodes by tag."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND tags LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.agent_id, f'%"{tag}"%', limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def recall_important(self, min_importance: float = 0.7, limit: int = 10) -> List[Episode]:
        """Recall important episodes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND importance >= ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        """, (self.agent_id, min_importance, limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def recall_positive(self, min_valence: float = 0.3, limit: int = 10) -> List[Episode]:
        """Recall positive episodes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND emotional_valence >= ?
            ORDER BY emotional_valence DESC, timestamp DESC
            LIMIT ?
        """, (self.agent_id, min_valence, limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def search(self, query: str, limit: int = 10) -> List[Episode]:
        """Search episodes by content."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM episodic_memory
            WHERE agent_id = ? AND (summary LIKE ? OR content LIKE ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.agent_id, f'%{query}%', f'%{query}%', limit))

        episodes = self._rows_to_episodes(cursor.fetchall())
        conn.close()
        return episodes

    def link_episodes(self, episode_id1: int, episode_id2: int):
        """Link two related episodes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current links for episode 1
        cursor.execute(
            "SELECT linked_episodes FROM episodic_memory WHERE id = ?",
            (episode_id1,)
        )
        row = cursor.fetchone()
        if row:
            links = json.loads(row['linked_episodes']) if row['linked_episodes'] else []
            if episode_id2 not in links:
                links.append(episode_id2)
                cursor.execute(
                    "UPDATE episodic_memory SET linked_episodes = ? WHERE id = ?",
                    (json.dumps(links), episode_id1)
                )

        # Get current links for episode 2
        cursor.execute(
            "SELECT linked_episodes FROM episodic_memory WHERE id = ?",
            (episode_id2,)
        )
        row = cursor.fetchone()
        if row:
            links = json.loads(row['linked_episodes']) if row['linked_episodes'] else []
            if episode_id1 not in links:
                links.append(episode_id1)
                cursor.execute(
                    "UPDATE episodic_memory SET linked_episodes = ? WHERE id = ?",
                    (json.dumps(links), episode_id2)
                )

        conn.commit()
        conn.close()

    def _rows_to_episodes(self, rows) -> List[Episode]:
        """Convert database rows to Episode objects."""
        episodes = []
        for row in rows:
            episodes.append(Episode(
                id=row['id'],
                episode_type=row['episode_type'],
                summary=row['summary'],
                content=row['content'],
                participants=json.loads(row['participants']) if row['participants'] else [],
                emotional_valence=row['emotional_valence'] or 0.0,
                importance=row['importance'],
                timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
                duration_seconds=row['duration_seconds'],
                location=row['location'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                linked_episodes=json.loads(row['linked_episodes']) if row['linked_episodes'] else [],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))
        return episodes

    def count(self) -> int:
        """Get total episode count."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodic_memory WHERE agent_id = ?", (self.agent_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_summary(self) -> str:
        """Get a summary of episodic memory for context injection."""
        recent = self.recall_recent(limit=3)
        important = self.recall_important(limit=2)

        summary_parts = []

        if recent:
            summary_parts.append("Recent Interactions:")
            for e in recent:
                time_str = e.timestamp.strftime("%Y-%m-%d %H:%M") if e.timestamp else "unknown time"
                summary_parts.append(f"  - [{time_str}] {e.summary}")

        if important:
            # Avoid duplicates
            recent_ids = {e.id for e in recent}
            important_unique = [e for e in important if e.id not in recent_ids]
            if important_unique:
                summary_parts.append("Notable Past Events:")
                for e in important_unique:
                    time_str = e.timestamp.strftime("%Y-%m-%d") if e.timestamp else "unknown"
                    summary_parts.append(f"  - [{time_str}] {e.summary}")

        return '\n'.join(summary_parts) if summary_parts else "No episodic memories recorded."
