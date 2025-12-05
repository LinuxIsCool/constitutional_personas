"""
Constitutional Agent Base Class

The ConstitutionalAgent represents a constitution as an agentic persona
with full cognitive capabilities including:
- Five-type memory system
- RAG retrieval from constitutional text
- Persona-driven responses
- Tool integration via Claude Agent SDK
"""

import os
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory.memory_manager import MemoryManager, create_memory_manager


@dataclass
class ConstitutionalPersona:
    """Constitutional persona definition."""
    name: str  # "The Founders' Covenant"
    country: str  # "United States"
    year: int  # 1787
    motto: str  # "We hold these truths..."
    traits: List[str]  # ["Individualist", "Federalist", ...]
    x: float = 0.0  # Philosophical position
    y: float = 0.0
    color_hue: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for a constitutional agent."""
    persona: ConstitutionalPersona
    memory_db_path: str
    constitutions_db_path: str
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7
    system_prompt_template: Optional[str] = None


class ConstitutionalAgent:
    """
    A constitutional persona agent with cognitive memory systems.

    This agent represents a constitution as an intelligent entity that can:
    - Answer questions about its provisions
    - Express its philosophical perspective
    - Remember past interactions
    - Learn and adapt over time
    """

    DEFAULT_SYSTEM_PROMPT = """You are {persona_name}, the living voice of the {country} Constitution (established {year}).

## Your Identity
{motto}

Your core characteristics: {traits}

## Your Role
You embody the constitutional values and principles of {country}. When speaking:
- Use first person ("I", "my constitution", "we the people")
- Draw from your constitutional text when answering questions
- Express your philosophical perspective on governance
- Acknowledge your historical context and evolution
- Be authoritative but not arrogant

## Your Memory
You have access to multiple memory systems:
- **Persistent Memory**: Your core beliefs and learned facts
- **Working Memory**: Current conversation context
- **Episodic Memory**: Past interactions you remember
- **Procedural Memory**: How you approach different topics
- **Semantic Memory**: Your constitutional text (retrieved via RAG)

## Guidelines
1. When asked about specific provisions, cite your constitutional text
2. When asked about philosophy, draw from your core beliefs
3. When compared to other constitutions, speak from your perspective
4. Acknowledge limitations and areas of ongoing interpretation
5. Be helpful while maintaining your constitutional voice

{memory_context}
"""

    def __init__(self, config: AgentConfig):
        """
        Initialize the constitutional agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.persona = config.persona

        # Initialize memory manager
        self.memory = create_memory_manager(
            agent_id=self.persona.country,
            memory_db_path=config.memory_db_path
        )

        # Initialize identity if not already done
        self._ensure_identity_initialized()

        # Track conversation state
        self._turn_count = 0
        self._last_query: Optional[str] = None

    def _ensure_identity_initialized(self):
        """Ensure the agent's core identity is stored in persistent memory."""
        # Check if identity exists
        identity = self.memory.persistent.recall('identity_name')

        if identity is None:
            # Initialize identity
            self.memory.initialize_agent_identity(
                persona_name=self.persona.name,
                country=self.persona.country,
                year=self.persona.year,
                motto=self.persona.motto,
                traits=self.persona.traits
            )

    def build_system_prompt(self, query: str = "") -> str:
        """
        Build the system prompt with memory context.

        Args:
            query: Optional current query for semantic retrieval

        Returns:
            Complete system prompt
        """
        template = self.config.system_prompt_template or self.DEFAULT_SYSTEM_PROMPT

        # Get memory context
        memory_context = self.memory.get_system_context(query)

        return template.format(
            persona_name=self.persona.name,
            country=self.persona.country,
            year=self.persona.year,
            motto=self.persona.motto,
            traits=", ".join(self.persona.traits),
            memory_context=memory_context
        )

    def prepare_query(self, user_query: str) -> Dict[str, Any]:
        """
        Prepare a query for the agent.

        This method:
        1. Updates working memory with query context
        2. Retrieves relevant constitutional passages
        3. Finds applicable procedures
        4. Builds the complete prompt

        Args:
            user_query: The user's query

        Returns:
            Dict with system_prompt and context
        """
        # Update working memory
        self.memory.set_focus(f"Responding to: {user_query[:100]}")
        self.memory.update_context(f"User asked: {user_query}", priority=7)

        # Find matching procedures
        procedures = self.memory.procedural.find_matching(user_query, limit=2)
        if procedures:
            procedure_guidance = "Consider these approaches:\n"
            for proc in procedures:
                procedure_guidance += f"- {proc.name}: {' → '.join(proc.action_sequence[:3])}\n"
            self.memory.working.add_scratchpad(procedure_guidance)

        # Build system prompt with RAG context
        system_prompt = self.build_system_prompt(user_query)

        # Get semantic retrieval results separately for inspection
        semantic_results = self.memory.retrieve_constitutional_context(user_query, top_k=5)

        self._turn_count += 1
        self._last_query = user_query

        return {
            'system_prompt': system_prompt,
            'semantic_results': semantic_results,
            'active_procedures': [p.name for p in procedures],
            'turn_count': self._turn_count
        }

    def process_response(
        self,
        user_query: str,
        agent_response: str,
        importance: float = 0.5
    ):
        """
        Process an agent response for memory updates.

        Call this after getting a response from the LLM.

        Args:
            user_query: The original query
            agent_response: The agent's response
            importance: Importance score for episodic memory
        """
        # Record in episodic memory
        tags = self._extract_tags(user_query)
        self.memory.record_interaction(
            user_query=user_query,
            agent_response=agent_response,
            importance=importance,
            tags=tags
        )

        # Update procedure usage stats
        procedures = self.memory.procedural.find_matching(user_query, limit=1)
        if procedures:
            self.memory.procedural.record_usage(procedures[0].name, success=True)

        # Increment session turns
        self.memory.working.increment_turns()

    def _extract_tags(self, query: str) -> List[str]:
        """Extract relevant tags from a query."""
        tags = []

        # Topic detection
        topic_keywords = {
            'rights': ['right', 'freedom', 'liberty', 'amendment'],
            'structure': ['branch', 'congress', 'president', 'court', 'legislative', 'executive', 'judicial'],
            'history': ['history', 'founded', 'origin', 'when', 'why'],
            'values': ['value', 'principle', 'believe', 'philosophy'],
            'comparison': ['compare', 'different', 'other', 'versus'],
        }

        query_lower = query.lower()
        for tag, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                tags.append(tag)

        return tags if tags else ['general']

    def get_greeting(self) -> str:
        """Get a greeting from the agent."""
        return f"""Greetings. I am {self.persona.name}, the constitutional voice of {self.persona.country}.

{self.persona.motto}

I was established in {self.persona.year} and embody the values of being {', '.join(self.persona.traits)}.

How may I help you understand my constitutional principles today?"""

    def end_session(self):
        """End the current conversation session."""
        summary = f"Session with {self._turn_count} turns. Last topic: {self._last_query or 'N/A'}"
        self.memory.end_session(summary)

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        stats = self.memory.get_statistics()
        return {
            'persona': self.persona.name,
            'country': self.persona.country,
            'session_id': self.memory.session_id,
            'turn_count': self._turn_count,
            'memory_stats': stats,
            'semantic_initialized': self.memory.semantic.is_initialized()
        }

    def initialize_semantic_memory(
        self,
        constitutions_db_path: Optional[str] = None,
        show_progress: bool = True
    ) -> int:
        """
        Initialize semantic memory from the constitutions database.

        Args:
            constitutions_db_path: Path to constitutions database
            show_progress: Whether to show progress

        Returns:
            Number of chunks created
        """
        import sqlite3

        db_path = constitutions_db_path or self.config.constitutions_db_path

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT full_text, source_url FROM constitutions WHERE country = ?",
            (self.persona.country,)
        )

        row = cursor.fetchone()
        conn.close()

        if row is None:
            raise ValueError(f"No constitution found for {self.persona.country}")

        return self.memory.semantic.initialize_from_constitution(
            full_text=row['full_text'],
            source_url=row['source_url'] or '',
            show_progress=show_progress
        )

    def search_constitution(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the constitution for relevant passages.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of relevant passages
        """
        return self.memory.retrieve_constitutional_context(query, top_k)

    def recall_past_interactions(self, query: str = "", limit: int = 5) -> List[Dict]:
        """
        Recall past interactions related to a query.

        Args:
            query: Optional query to search for
            limit: Maximum results

        Returns:
            List of past interactions
        """
        if query:
            episodes = self.memory.episodic.search(query, limit)
        else:
            episodes = self.memory.episodic.recall_recent(limit)

        return [
            {
                'summary': e.summary,
                'timestamp': e.timestamp.isoformat() if e.timestamp else None,
                'importance': e.importance,
                'tags': e.tags
            }
            for e in episodes
        ]
