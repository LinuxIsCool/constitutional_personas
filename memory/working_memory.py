"""
Working Memory for Constitutional Agents

Current session context, active goals, and temporary information.
This memory is session-scoped and provides the "scratchpad" for
active reasoning and goal tracking.

Examples:
- Goals: "Help user understand the Bill of Rights"
- Context: "User is asking about freedom of speech"
- Focus: "Currently discussing First Amendment"
- Scratchpad: "Need to mention landmark cases"
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WorkingMemoryEntry:
    """A working memory entry."""
    id: int
    session_id: str
    memory_type: str  # goal, context, focus, scratchpad
    content: str
    priority: int
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    parent_id: Optional[int]
    metadata: Dict[str, Any]


class WorkingMemory:
    """
    Working memory store for current session context.

    This memory type is designed for:
    - Active goals and subgoals (hierarchical)
    - Current conversation context
    - Temporary scratchpad notes
    - Focus tracking for multi-turn conversations
    """

    def __init__(self, db_path: str, agent_id: str, session_id: Optional[str] = None):
        """
        Initialize working memory.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent
            session_id: Session ID (generated if not provided)
        """
        self.db_path = db_path
        self.agent_id = agent_id
        self.session_id = session_id or str(uuid.uuid4())
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure the working_memory and agent_sessions tables exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                parent_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (parent_id) REFERENCES working_memory(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                agent_id TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                total_turns INTEGER DEFAULT 0,
                summary TEXT,
                is_active BOOLEAN DEFAULT 1,
                metadata TEXT
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_working_session ON working_memory(agent_id, session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_working_active ON working_memory(agent_id, is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent ON agent_sessions(agent_id)")

        conn.commit()
        conn.close()

        # Register this session
        self._register_session()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _register_session(self):
        """Register this session in the sessions table."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR IGNORE INTO agent_sessions (session_id, agent_id)
            VALUES (?, ?)
        """, (self.session_id, self.agent_id))

        conn.commit()
        conn.close()

    def add_goal(
        self,
        goal: str,
        priority: int = 5,
        parent_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a goal to working memory.

        Args:
            goal: The goal description
            priority: Priority 1-10 (higher = more important)
            parent_id: Parent goal ID for subgoals
            metadata: Additional metadata

        Returns:
            The goal ID
        """
        return self._add_entry('goal', goal, priority, parent_id, metadata)

    def add_context(
        self,
        context: str,
        priority: int = 5,
        expires_minutes: Optional[int] = 60,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add context information to working memory.

        Args:
            context: The context information
            priority: Priority 1-10
            expires_minutes: How long until this context expires
            metadata: Additional metadata

        Returns:
            The context entry ID
        """
        expires_at = None
        if expires_minutes:
            expires_at = (datetime.now() + timedelta(minutes=expires_minutes)).isoformat()

        return self._add_entry('context', context, priority, None, metadata, expires_at)

    def add_focus(
        self,
        focus: str,
        priority: int = 8,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Set the current focus of attention.

        Higher priority than most other entries.
        """
        # Deactivate previous focus entries
        self._deactivate_type('focus')
        return self._add_entry('focus', focus, priority, None, metadata)

    def add_scratchpad(
        self,
        note: str,
        priority: int = 3,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a scratchpad note."""
        return self._add_entry('scratchpad', note, priority, None, metadata)

    def _add_entry(
        self,
        memory_type: str,
        content: str,
        priority: int,
        parent_id: Optional[int],
        metadata: Optional[Dict[str, Any]],
        expires_at: Optional[str] = None
    ) -> int:
        """Add an entry to working memory."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO working_memory (
                agent_id, session_id, memory_type, content,
                priority, parent_id, metadata, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.agent_id, self.session_id, memory_type, content,
            priority, parent_id, json.dumps(metadata) if metadata else None,
            expires_at
        ))

        entry_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return entry_id

    def _deactivate_type(self, memory_type: str):
        """Deactivate all entries of a specific type."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE working_memory
            SET is_active = 0
            WHERE agent_id = ? AND session_id = ? AND memory_type = ?
        """, (self.agent_id, self.session_id, memory_type))

        conn.commit()
        conn.close()

    def get_active(self, memory_type: Optional[str] = None) -> List[WorkingMemoryEntry]:
        """
        Get active working memory entries.

        Args:
            memory_type: Filter by type (goal, context, focus, scratchpad)

        Returns:
            List of active entries sorted by priority
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.now().isoformat()

        if memory_type:
            cursor.execute("""
                SELECT * FROM working_memory
                WHERE agent_id = ? AND session_id = ? AND memory_type = ?
                AND is_active = 1
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY priority DESC, created_at DESC
            """, (self.agent_id, self.session_id, memory_type, now))
        else:
            cursor.execute("""
                SELECT * FROM working_memory
                WHERE agent_id = ? AND session_id = ?
                AND is_active = 1
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY priority DESC, created_at DESC
            """, (self.agent_id, self.session_id, now))

        entries = []
        for row in cursor.fetchall():
            entries.append(WorkingMemoryEntry(
                id=row['id'],
                session_id=row['session_id'],
                memory_type=row['memory_type'],
                content=row['content'],
                priority=row['priority'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                is_active=bool(row['is_active']),
                parent_id=row['parent_id'],
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            ))

        conn.close()
        return entries

    def get_goals(self) -> List[WorkingMemoryEntry]:
        """Get all active goals."""
        return self.get_active('goal')

    def get_context(self) -> List[WorkingMemoryEntry]:
        """Get all active context entries."""
        return self.get_active('context')

    def get_focus(self) -> Optional[WorkingMemoryEntry]:
        """Get the current focus."""
        entries = self.get_active('focus')
        return entries[0] if entries else None

    def get_scratchpad(self) -> List[WorkingMemoryEntry]:
        """Get all scratchpad notes."""
        return self.get_active('scratchpad')

    def complete_goal(self, goal_id: int):
        """Mark a goal as completed (deactivate it)."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE working_memory
            SET is_active = 0
            WHERE id = ? AND agent_id = ? AND memory_type = 'goal'
        """, (goal_id, self.agent_id))

        conn.commit()
        conn.close()

    def clear_scratchpad(self):
        """Clear all scratchpad notes."""
        self._deactivate_type('scratchpad')

    def clear_context(self):
        """Clear all context entries."""
        self._deactivate_type('context')

    def clear_all(self):
        """Clear all working memory for this session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE working_memory
            SET is_active = 0
            WHERE agent_id = ? AND session_id = ?
        """, (self.agent_id, self.session_id))

        conn.commit()
        conn.close()

    def increment_turns(self):
        """Increment the turn counter for this session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE agent_sessions
            SET total_turns = total_turns + 1
            WHERE session_id = ?
        """, (self.session_id,))

        conn.commit()
        conn.close()

    def end_session(self, summary: Optional[str] = None):
        """End this session."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE agent_sessions
            SET is_active = 0, ended_at = ?, summary = ?
            WHERE session_id = ?
        """, (datetime.now().isoformat(), summary, self.session_id))

        # Deactivate all working memory
        cursor.execute("""
            UPDATE working_memory
            SET is_active = 0
            WHERE agent_id = ? AND session_id = ?
        """, (self.agent_id, self.session_id))

        conn.commit()
        conn.close()

    def get_summary(self) -> str:
        """Get a summary of working memory for context injection."""
        focus = self.get_focus()
        goals = self.get_goals()
        context = self.get_context()
        scratchpad = self.get_scratchpad()

        summary_parts = []

        if focus:
            summary_parts.append(f"Current Focus: {focus.content}")

        if goals:
            summary_parts.append("Active Goals:")
            for g in goals[:5]:  # Limit to 5
                summary_parts.append(f"  - {g.content}")

        if context:
            summary_parts.append("Context:")
            for c in context[:3]:  # Limit to 3
                summary_parts.append(f"  - {c.content}")

        if scratchpad:
            summary_parts.append("Notes:")
            for s in scratchpad[:3]:  # Limit to 3
                summary_parts.append(f"  - {s.content}")

        return '\n'.join(summary_parts) if summary_parts else "Working memory is empty."
