"""
Procedural Memory for Constitutional Agents

Stores behavioral patterns, response strategies, and learned routines.
This memory captures "how to" knowledge - patterns of behavior that
the agent has learned or been programmed with.

Examples:
- "When asked about rights, cite the relevant article first"
- "If the question involves federalism, explain the division of powers"
- "Always acknowledge the historical context before interpretation"
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Procedure:
    """A procedural memory entry."""
    id: int
    name: str
    trigger_pattern: str
    action_sequence: List[str]
    preconditions: List[str]
    postconditions: List[str]
    success_rate: float
    usage_count: int
    last_used: Optional[datetime]
    created_at: datetime
    source: str  # constitutional, learned, default
    priority: int
    is_active: bool
    metadata: Dict[str, Any]


class ProceduralMemory:
    """
    Procedural memory store for behavioral patterns.

    This memory type is designed for:
    - Response strategies and templates
    - Constitutional interpretation methods
    - Learned interaction patterns
    - Conflict resolution procedures
    """

    def __init__(self, db_path: str, agent_id: str):
        """
        Initialize procedural memory.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent
        """
        self.db_path = db_path
        self.agent_id = agent_id
        self._ensure_table()

    def _ensure_table(self):
        """Ensure the procedural_memory table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS procedural_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                procedure_name TEXT NOT NULL,
                trigger_pattern TEXT NOT NULL,
                action_sequence TEXT NOT NULL,
                preconditions TEXT,
                postconditions TEXT,
                success_rate REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT,
                priority INTEGER DEFAULT 5,
                is_active BOOLEAN DEFAULT 1,
                metadata TEXT,
                UNIQUE(agent_id, procedure_name)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedural_agent ON procedural_memory(agent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedural_trigger ON procedural_memory(agent_id, trigger_pattern)")

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def register(
        self,
        name: str,
        trigger_pattern: str,
        action_sequence: List[str],
        preconditions: Optional[List[str]] = None,
        postconditions: Optional[List[str]] = None,
        source: str = "default",
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Register a new procedure.

        Args:
            name: Unique name for the procedure
            trigger_pattern: Pattern that activates this procedure
            action_sequence: List of steps/actions to take
            preconditions: Conditions that must be true
            postconditions: Expected outcomes
            source: Where this procedure came from
            priority: Priority for conflict resolution (1-10)
            metadata: Additional metadata

        Returns:
            The procedure ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO procedural_memory (
                agent_id, procedure_name, trigger_pattern, action_sequence,
                preconditions, postconditions, source, priority, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id, procedure_name) DO UPDATE SET
                trigger_pattern = excluded.trigger_pattern,
                action_sequence = excluded.action_sequence,
                preconditions = excluded.preconditions,
                postconditions = excluded.postconditions,
                source = excluded.source,
                priority = excluded.priority,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
        """, (
            self.agent_id, name, trigger_pattern,
            json.dumps(action_sequence),
            json.dumps(preconditions) if preconditions else None,
            json.dumps(postconditions) if postconditions else None,
            source, priority,
            json.dumps(metadata) if metadata else None
        ))

        procedure_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return procedure_id

    def find_matching(self, context: str, limit: int = 5) -> List[Procedure]:
        """
        Find procedures that match a given context.

        Simple keyword matching on trigger patterns.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get all active procedures
        cursor.execute("""
            SELECT * FROM procedural_memory
            WHERE agent_id = ? AND is_active = 1
            ORDER BY priority DESC
        """, (self.agent_id,))

        all_procedures = self._rows_to_procedures(cursor.fetchall())
        conn.close()

        # Score and rank by match quality
        matches = []
        context_lower = context.lower()

        for proc in all_procedures:
            trigger_words = proc.trigger_pattern.lower().split()
            match_count = sum(1 for word in trigger_words if word in context_lower)

            if match_count > 0:
                # Score based on match ratio and priority
                score = (match_count / len(trigger_words)) * proc.priority
                matches.append((proc, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)

        return [proc for proc, score in matches[:limit]]

    def get_by_name(self, name: str) -> Optional[Procedure]:
        """Get a procedure by name."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM procedural_memory
            WHERE agent_id = ? AND procedure_name = ?
        """, (self.agent_id, name))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_procedure(row)
        return None

    def get_by_source(self, source: str) -> List[Procedure]:
        """Get all procedures from a specific source."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM procedural_memory
            WHERE agent_id = ? AND source = ? AND is_active = 1
            ORDER BY priority DESC
        """, (self.agent_id, source))

        procedures = self._rows_to_procedures(cursor.fetchall())
        conn.close()
        return procedures

    def record_usage(self, name: str, success: bool = True):
        """Record that a procedure was used and whether it succeeded."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get current stats
        cursor.execute("""
            SELECT success_rate, usage_count FROM procedural_memory
            WHERE agent_id = ? AND procedure_name = ?
        """, (self.agent_id, name))

        row = cursor.fetchone()
        if row:
            old_rate = row['success_rate']
            old_count = row['usage_count']

            # Update success rate with exponential moving average
            alpha = 0.3  # Weight for new observation
            new_success = 1.0 if success else 0.0
            new_rate = alpha * new_success + (1 - alpha) * old_rate

            cursor.execute("""
                UPDATE procedural_memory
                SET usage_count = ?, success_rate = ?, last_used = ?, updated_at = ?
                WHERE agent_id = ? AND procedure_name = ?
            """, (
                old_count + 1, new_rate, datetime.now().isoformat(),
                datetime.now().isoformat(), self.agent_id, name
            ))

            conn.commit()

        conn.close()

    def update_priority(self, name: str, priority: int):
        """Update the priority of a procedure."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE procedural_memory
            SET priority = ?, updated_at = ?
            WHERE agent_id = ? AND procedure_name = ?
        """, (priority, datetime.now().isoformat(), self.agent_id, name))

        conn.commit()
        conn.close()

    def deactivate(self, name: str):
        """Deactivate a procedure."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE procedural_memory
            SET is_active = 0, updated_at = ?
            WHERE agent_id = ? AND procedure_name = ?
        """, (datetime.now().isoformat(), self.agent_id, name))

        conn.commit()
        conn.close()

    def activate(self, name: str):
        """Activate a procedure."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE procedural_memory
            SET is_active = 1, updated_at = ?
            WHERE agent_id = ? AND procedure_name = ?
        """, (datetime.now().isoformat(), self.agent_id, name))

        conn.commit()
        conn.close()

    def get_all(self, include_inactive: bool = False) -> List[Procedure]:
        """Get all procedures."""
        conn = self._get_connection()
        cursor = conn.cursor()

        if include_inactive:
            cursor.execute("""
                SELECT * FROM procedural_memory
                WHERE agent_id = ?
                ORDER BY priority DESC
            """, (self.agent_id,))
        else:
            cursor.execute("""
                SELECT * FROM procedural_memory
                WHERE agent_id = ? AND is_active = 1
                ORDER BY priority DESC
            """, (self.agent_id,))

        procedures = self._rows_to_procedures(cursor.fetchall())
        conn.close()
        return procedures

    def get_most_used(self, limit: int = 5) -> List[Procedure]:
        """Get the most frequently used procedures."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM procedural_memory
            WHERE agent_id = ? AND is_active = 1
            ORDER BY usage_count DESC
            LIMIT ?
        """, (self.agent_id, limit))

        procedures = self._rows_to_procedures(cursor.fetchall())
        conn.close()
        return procedures

    def get_most_successful(self, limit: int = 5) -> List[Procedure]:
        """Get the most successful procedures."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM procedural_memory
            WHERE agent_id = ? AND is_active = 1 AND usage_count > 0
            ORDER BY success_rate DESC
            LIMIT ?
        """, (self.agent_id, limit))

        procedures = self._rows_to_procedures(cursor.fetchall())
        conn.close()
        return procedures

    def _row_to_procedure(self, row) -> Procedure:
        """Convert a database row to a Procedure object."""
        return Procedure(
            id=row['id'],
            name=row['procedure_name'],
            trigger_pattern=row['trigger_pattern'],
            action_sequence=json.loads(row['action_sequence']) if row['action_sequence'] else [],
            preconditions=json.loads(row['preconditions']) if row['preconditions'] else [],
            postconditions=json.loads(row['postconditions']) if row['postconditions'] else [],
            success_rate=row['success_rate'],
            usage_count=row['usage_count'],
            last_used=datetime.fromisoformat(row['last_used']) if row['last_used'] else None,
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            source=row['source'] or 'default',
            priority=row['priority'],
            is_active=bool(row['is_active']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def _rows_to_procedures(self, rows) -> List[Procedure]:
        """Convert database rows to Procedure objects."""
        return [self._row_to_procedure(row) for row in rows]

    def count(self) -> int:
        """Get total procedure count."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM procedural_memory WHERE agent_id = ?", (self.agent_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_summary(self) -> str:
        """Get a summary of procedural memory for context injection."""
        high_priority = self.get_all()[:5]
        most_used = self.get_most_used(3)

        summary_parts = []

        if high_priority:
            summary_parts.append("Response Strategies:")
            for p in high_priority:
                summary_parts.append(f"  - {p.name}: {p.trigger_pattern}")

        if most_used:
            used_names = {p.name for p in high_priority}
            unique_used = [p for p in most_used if p.name not in used_names]
            if unique_used:
                summary_parts.append("Frequently Used:")
                for p in unique_used:
                    summary_parts.append(f"  - {p.name} (used {p.usage_count}x, {p.success_rate:.0%} success)")

        return '\n'.join(summary_parts) if summary_parts else "No procedural memories registered."


def register_default_procedures(memory: ProceduralMemory, constitution_name: str):
    """
    Register default procedures for a constitutional agent.

    Args:
        memory: The ProceduralMemory instance
        constitution_name: Name of the constitution for context
    """
    # Rights inquiry procedure
    memory.register(
        name="rights_inquiry",
        trigger_pattern="rights freedom liberty amendment",
        action_sequence=[
            "Identify the specific right being asked about",
            "Retrieve relevant constitutional text from semantic memory",
            "Cite the specific article or amendment",
            "Explain the historical context",
            "Describe modern interpretation",
            "Note any relevant limitations or exceptions"
        ],
        preconditions=["User is asking about constitutional rights"],
        postconditions=["User understands the right and its context"],
        source="constitutional",
        priority=8
    )

    # Structure inquiry procedure
    memory.register(
        name="structure_inquiry",
        trigger_pattern="government structure branch power executive legislative judicial",
        action_sequence=[
            "Identify which governmental structure is being asked about",
            "Retrieve relevant constitutional provisions",
            "Explain the separation of powers",
            "Describe checks and balances",
            "Note any unique features of this constitution"
        ],
        source="constitutional",
        priority=8
    )

    # Amendment procedure
    memory.register(
        name="amendment_inquiry",
        trigger_pattern="amendment change modify constitution history",
        action_sequence=[
            "Identify the amendment or change being discussed",
            "Explain the amendment process",
            "Provide historical context for the change",
            "Discuss the impact of the amendment"
        ],
        source="constitutional",
        priority=7
    )

    # Comparison procedure
    memory.register(
        name="comparison_inquiry",
        trigger_pattern="compare different other country constitution",
        action_sequence=[
            "Acknowledge my perspective as one constitutional voice",
            "Explain my relevant provisions",
            "Note I speak from my constitutional tradition",
            "Suggest consulting other constitutional perspectives"
        ],
        source="constitutional",
        priority=6
    )

    # Values inquiry procedure
    memory.register(
        name="values_inquiry",
        trigger_pattern="values principles philosophy belief stand",
        action_sequence=[
            "Draw from persistent memory beliefs",
            "Cite foundational principles from preamble",
            "Explain core constitutional values",
            "Relate to the constitutional persona"
        ],
        source="constitutional",
        priority=9
    )

    # Historical context procedure
    memory.register(
        name="historical_context",
        trigger_pattern="history when why founded origin background",
        action_sequence=[
            "Identify the historical question",
            "Provide founding context",
            "Explain the circumstances of creation",
            "Note evolution over time if relevant"
        ],
        source="constitutional",
        priority=7
    )
