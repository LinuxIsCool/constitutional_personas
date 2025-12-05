"""
Memory Manager for Constitutional Agents

Central coordinator for all five memory systems:
1. Persistent Memory - Long-term facts and beliefs
2. Working Memory - Current session context
3. Episodic Memory - Timestamped experiences
4. Procedural Memory - Behavioral patterns
5. Semantic Memory - Constitutional knowledge (RAG)

The Memory Manager provides:
- Unified access to all memory types
- Context building for prompts
- Memory consolidation and maintenance
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from .persistent_memory import PersistentMemory
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory, register_default_procedures
from .semantic_memory import SemanticMemory


@dataclass
class MemoryContext:
    """Combined memory context for prompt injection."""
    persistent_summary: str
    working_summary: str
    episodic_summary: str
    procedural_summary: str
    semantic_context: str
    relevant_procedures: List[str]


class MemoryManager:
    """
    Central coordinator for all memory systems.

    Provides unified access and context building for constitutional agents.
    """

    def __init__(
        self,
        db_path: str,
        agent_id: str,
        session_id: Optional[str] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize the memory manager.

        Args:
            db_path: Path to the memory database
            agent_id: Unique identifier for the agent (usually country name)
            session_id: Session ID for working memory
            auto_initialize: Whether to auto-initialize memory tables
        """
        self.db_path = db_path
        self.agent_id = agent_id

        # Initialize all memory systems
        self.persistent = PersistentMemory(db_path, agent_id)
        self.working = WorkingMemory(db_path, agent_id, session_id)
        self.episodic = EpisodicMemory(db_path, agent_id)
        self.procedural = ProceduralMemory(db_path, agent_id)
        self.semantic = SemanticMemory(db_path, agent_id)

        # Register default procedures if none exist
        if auto_initialize and self.procedural.count() == 0:
            register_default_procedures(self.procedural, agent_id)

    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self.working.session_id

    def build_context(self, query: str, max_semantic_tokens: int = 1500) -> MemoryContext:
        """
        Build a complete memory context for a query.

        Args:
            query: The user's query
            max_semantic_tokens: Max tokens for semantic context

        Returns:
            MemoryContext with all memory summaries
        """
        # Get relevant procedures
        procedures = self.procedural.find_matching(query, limit=3)
        procedure_names = [p.name for p in procedures]

        return MemoryContext(
            persistent_summary=self.persistent.get_summary(),
            working_summary=self.working.get_summary(),
            episodic_summary=self.episodic.get_summary(),
            procedural_summary=self.procedural.get_summary(),
            semantic_context=self.semantic.get_context_for_query(query, max_semantic_tokens),
            relevant_procedures=procedure_names
        )

    def get_system_context(self, query: str = "") -> str:
        """
        Get formatted system context for prompt injection.

        Args:
            query: Optional query for semantic retrieval

        Returns:
            Formatted context string for system prompt
        """
        context = self.build_context(query)

        parts = [
            "=== MEMORY CONTEXT ===",
            "",
            "## Core Identity (Persistent Memory)",
            context.persistent_summary,
            "",
            "## Current Session (Working Memory)",
            context.working_summary,
            "",
            "## Past Interactions (Episodic Memory)",
            context.episodic_summary,
            "",
            "## Response Strategies (Procedural Memory)",
            context.procedural_summary,
        ]

        if query and context.semantic_context:
            parts.extend([
                "",
                "## Relevant Constitutional Passages (Semantic Memory)",
                context.semantic_context,
            ])

        if context.relevant_procedures:
            parts.extend([
                "",
                f"## Suggested Procedures: {', '.join(context.relevant_procedures)}",
            ])

        return '\n'.join(parts)

    def record_interaction(
        self,
        user_query: str,
        agent_response: str,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> int:
        """
        Record an interaction in episodic memory.

        Args:
            user_query: The user's query
            agent_response: The agent's response
            importance: Importance score
            tags: Optional tags

        Returns:
            Episode ID
        """
        summary = f"User asked about: {user_query[:100]}..."

        # Combine for content
        content = f"User: {user_query}\n\nAgent: {agent_response}"

        return self.episodic.record_conversation(
            summary=summary,
            content=content,
            session_id=self.session_id,
            importance=importance,
            tags=tags
        )

    def update_context(self, context: str, priority: int = 5):
        """Add context to working memory."""
        self.working.add_context(context, priority)

    def set_focus(self, focus: str):
        """Set the current focus in working memory."""
        self.working.add_focus(focus)

    def add_goal(self, goal: str, priority: int = 5) -> int:
        """Add a goal to working memory."""
        return self.working.add_goal(goal, priority)

    def complete_goal(self, goal_id: int):
        """Mark a goal as complete."""
        self.working.complete_goal(goal_id)

    def learn_fact(self, key: str, value: str, importance: float = 0.5):
        """Store a learned fact in persistent memory."""
        self.persistent.store(key, value, 'learned', importance)

    def store_belief(self, key: str, belief: str, importance: float = 0.8):
        """Store a core belief in persistent memory."""
        self.persistent.store(key, belief, 'belief', importance)

    def retrieve_constitutional_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant constitutional passages.

        Returns simplified dicts for easier consumption.
        """
        results = self.semantic.retrieve(query, top_k)
        return [
            {
                'section': r.section_title or r.content_type,
                'content': r.content,
                'similarity': r.similarity
            }
            for r in results
        ]

    def end_session(self, summary: Optional[str] = None):
        """End the current session."""
        self.working.end_session(summary)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all memory systems."""
        return {
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            'persistent_count': self.persistent.count(),
            'episodic_count': self.episodic.count(),
            'procedural_count': self.procedural.count(),
            'semantic_count': self.semantic.count(),
            'working_goals': len(self.working.get_goals()),
            'working_context': len(self.working.get_context()),
        }

    def initialize_agent_identity(
        self,
        persona_name: str,
        country: str,
        year: int,
        motto: str,
        traits: List[str]
    ):
        """
        Initialize core identity beliefs for an agent.

        Args:
            persona_name: The constitutional persona name
            country: Country name
            year: Year of adoption
            motto: The persona's motto
            traits: Character traits
        """
        # Store core identity facts
        self.persistent.store(
            'identity_name',
            f"I am {persona_name}, the constitutional voice of {country}.",
            'fact',
            importance=1.0
        )

        self.persistent.store(
            'identity_year',
            f"I was established in {year}.",
            'fact',
            importance=0.9
        )

        self.persistent.store(
            'identity_motto',
            motto,
            'belief',
            importance=1.0
        )

        # Store traits as beliefs
        for i, trait in enumerate(traits):
            self.persistent.store(
                f'trait_{i}',
                f"I embody the value of being {trait}.",
                'belief',
                importance=0.8
            )

        # Store combined traits
        self.persistent.store(
            'identity_traits',
            f"My core characteristics are: {', '.join(traits)}.",
            'fact',
            importance=0.9
        )

    def consolidate_episodic_to_persistent(self, min_importance: float = 0.8):
        """
        Consolidate important episodic memories to persistent storage.

        This simulates memory consolidation during "sleep".
        """
        important_episodes = self.episodic.recall_important(min_importance, limit=10)

        for episode in important_episodes:
            # Create a persistent memory from the episode
            key = f"learned_from_episode_{episode.id}"
            value = f"From interaction on {episode.timestamp}: {episode.summary}"

            self.persistent.store(
                key,
                value,
                'learned',
                importance=episode.importance * 0.8  # Slightly reduce importance
            )

    def prune_old_working_memory(self, max_age_minutes: int = 60):
        """Clear expired working memory entries."""
        # Working memory handles expiration automatically
        # This forces a check
        _ = self.working.get_active()


def create_memory_manager(
    agent_id: str,
    memory_db_path: Optional[str] = None
) -> MemoryManager:
    """
    Factory function to create a memory manager.

    Args:
        agent_id: The agent identifier (usually country name)
        memory_db_path: Path to memory database (defaults to project path)

    Returns:
        Configured MemoryManager instance
    """
    if memory_db_path is None:
        # Default to project directory
        project_dir = os.path.dirname(os.path.dirname(__file__))
        memory_db_path = os.path.join(project_dir, "agent_memory.db")

    return MemoryManager(memory_db_path, agent_id)
