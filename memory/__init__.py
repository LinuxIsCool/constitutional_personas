"""
Memory Module for Constitutional Agents

Implements a five-memory cognitive architecture:
1. PersistentMemory - Long-term facts, beliefs, preferences
2. WorkingMemory - Current session context and goals
3. EpisodicMemory - Timestamped experiences and interactions
4. ProceduralMemory - Behavioral patterns and response strategies
5. SemanticMemory - Constitutional knowledge with RAG retrieval

Each memory type is optimized for different cognitive functions,
enabling agents to maintain rich, contextualized understanding.
"""

from .persistent_memory import PersistentMemory
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .procedural_memory import ProceduralMemory
from .semantic_memory import SemanticMemory
from .memory_manager import MemoryManager

__all__ = [
    'PersistentMemory',
    'WorkingMemory',
    'EpisodicMemory',
    'ProceduralMemory',
    'SemanticMemory',
    'MemoryManager'
]
