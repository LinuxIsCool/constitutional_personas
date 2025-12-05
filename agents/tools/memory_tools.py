"""
Memory Tools for Constitutional Agents

These tools enable agents to interact with their memory systems.
Designed to work with the Claude Agent SDK @tool decorator pattern.

Usage with Claude Agent SDK:
    from claude_agent_sdk import tool, create_sdk_mcp_server
    from agents.tools.memory_tools import create_memory_tools

    tools = create_memory_tools(memory_manager)
    server = create_sdk_mcp_server(name="memory", tools=tools)
"""

import json
from typing import Dict, Any, List, Optional, Callable
from functools import wraps

# Import memory types for type hints
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memory.memory_manager import MemoryManager


def create_tool_response(content: str, is_error: bool = False) -> Dict:
    """Create a standard tool response."""
    return {
        "content": [{"type": "text", "text": content}],
        "is_error": is_error
    }


# Tool definitions following Claude Agent SDK pattern
# These can be used with @tool decorator or registered directly

def recall_memory(memory: MemoryManager, key: str) -> Dict:
    """
    Recall a specific memory by key.

    Args:
        memory: The memory manager instance
        key: The memory key to recall

    Returns:
        Tool response with memory content
    """
    entry = memory.persistent.recall(key)

    if entry:
        result = {
            "key": entry.key,
            "value": entry.value,
            "type": entry.memory_type,
            "importance": entry.importance,
            "confidence": entry.confidence
        }
        return create_tool_response(json.dumps(result, indent=2))
    else:
        return create_tool_response(f"No memory found for key: {key}", is_error=True)


def store_memory(
    memory: MemoryManager,
    key: str,
    value: str,
    memory_type: str = "learned",
    importance: float = 0.5
) -> Dict:
    """
    Store a new memory.

    Args:
        memory: The memory manager instance
        key: Unique key for the memory
        value: The content to store
        memory_type: Type (fact, belief, preference, learned)
        importance: Importance score 0.0-1.0

    Returns:
        Tool response confirming storage
    """
    try:
        memory.persistent.store(
            key=key,
            value=value,
            memory_type=memory_type,
            importance=importance
        )
        return create_tool_response(f"Memory stored successfully: {key}")
    except Exception as e:
        return create_tool_response(f"Error storing memory: {e}", is_error=True)


def search_memories(memory: MemoryManager, query: str, limit: int = 5) -> Dict:
    """
    Search across memories.

    Args:
        memory: The memory manager instance
        query: Search query
        limit: Maximum results

    Returns:
        Tool response with search results
    """
    # Search persistent memory
    persistent_results = memory.persistent.search(query, limit)

    # Search episodic memory
    episodic_results = memory.episodic.search(query, limit)

    results = {
        "persistent": [
            {
                "key": e.key,
                "value": e.value,
                "type": e.memory_type,
                "importance": e.importance
            }
            for e in persistent_results
        ],
        "episodic": [
            {
                "summary": e.summary,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "importance": e.importance
            }
            for e in episodic_results
        ]
    }

    return create_tool_response(json.dumps(results, indent=2))


def get_memory_summary(memory: MemoryManager) -> Dict:
    """
    Get a summary of all memory systems.

    Args:
        memory: The memory manager instance

    Returns:
        Tool response with memory summaries
    """
    summary = {
        "persistent": memory.persistent.get_summary(),
        "working": memory.working.get_summary(),
        "episodic": memory.episodic.get_summary(),
        "procedural": memory.procedural.get_summary(),
        "semantic": memory.semantic.get_summary(),
        "statistics": memory.get_statistics()
    }

    return create_tool_response(json.dumps(summary, indent=2))


def add_goal(memory: MemoryManager, goal: str, priority: int = 5) -> Dict:
    """
    Add a goal to working memory.

    Args:
        memory: The memory manager instance
        goal: The goal description
        priority: Priority 1-10

    Returns:
        Tool response with goal ID
    """
    goal_id = memory.add_goal(goal, priority)
    return create_tool_response(f"Goal added with ID: {goal_id}")


def complete_goal(memory: MemoryManager, goal_id: int) -> Dict:
    """
    Mark a goal as completed.

    Args:
        memory: The memory manager instance
        goal_id: The goal ID to complete

    Returns:
        Tool response confirming completion
    """
    memory.complete_goal(goal_id)
    return create_tool_response(f"Goal {goal_id} marked as complete")


def record_reflection(
    memory: MemoryManager,
    summary: str,
    content: str,
    importance: float = 0.6
) -> Dict:
    """
    Record a reflection in episodic memory.

    Args:
        memory: The memory manager instance
        summary: Brief summary
        content: Full reflection content
        importance: Importance score

    Returns:
        Tool response with episode ID
    """
    episode_id = memory.episodic.record_reflection(
        summary=summary,
        content=content,
        importance=importance
    )
    return create_tool_response(f"Reflection recorded with ID: {episode_id}")


def get_procedures(memory: MemoryManager, context: str) -> Dict:
    """
    Get relevant procedures for a context.

    Args:
        memory: The memory manager instance
        context: The context to match

    Returns:
        Tool response with matching procedures
    """
    procedures = memory.procedural.find_matching(context, limit=3)

    results = [
        {
            "name": p.name,
            "trigger": p.trigger_pattern,
            "actions": p.action_sequence,
            "success_rate": p.success_rate,
            "usage_count": p.usage_count
        }
        for p in procedures
    ]

    return create_tool_response(json.dumps(results, indent=2))


# Tool schema definitions for Claude Agent SDK
MEMORY_TOOL_SCHEMAS = {
    "recall_memory": {
        "name": "recall_memory",
        "description": "Recall a specific memory by its key from persistent memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The unique key of the memory to recall"
                }
            },
            "required": ["key"]
        }
    },
    "store_memory": {
        "name": "store_memory",
        "description": "Store a new fact, belief, or learned information in persistent memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Unique key for this memory"
                },
                "value": {
                    "type": "string",
                    "description": "The content to store"
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["fact", "belief", "preference", "learned"],
                    "description": "Type of memory",
                    "default": "learned"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0",
                    "default": 0.5
                }
            },
            "required": ["key", "value"]
        }
    },
    "search_memories": {
        "name": "search_memories",
        "description": "Search across persistent and episodic memories",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    "get_memory_summary": {
        "name": "get_memory_summary",
        "description": "Get a summary of all memory systems",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    "add_goal": {
        "name": "add_goal",
        "description": "Add a goal to working memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The goal description"
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority 1-10 (higher = more important)",
                    "default": 5
                }
            },
            "required": ["goal"]
        }
    },
    "complete_goal": {
        "name": "complete_goal",
        "description": "Mark a goal as completed",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal_id": {
                    "type": "integer",
                    "description": "The goal ID to complete"
                }
            },
            "required": ["goal_id"]
        }
    },
    "record_reflection": {
        "name": "record_reflection",
        "description": "Record a reflection or insight in episodic memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the reflection"
                },
                "content": {
                    "type": "string",
                    "description": "Full reflection content"
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0",
                    "default": 0.6
                }
            },
            "required": ["summary", "content"]
        }
    },
    "get_procedures": {
        "name": "get_procedures",
        "description": "Get relevant procedural strategies for a given context",
        "input_schema": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The context or topic to find procedures for"
                }
            },
            "required": ["context"]
        }
    }
}


def create_memory_tools(memory_manager: MemoryManager) -> List[Callable]:
    """
    Create memory tools bound to a specific memory manager.

    This creates closures that can be used with the Claude Agent SDK.

    Args:
        memory_manager: The MemoryManager instance to bind to

    Returns:
        List of tool functions ready for SDK registration
    """

    def _recall_memory(key: str) -> Dict:
        return recall_memory(memory_manager, key)

    def _store_memory(
        key: str,
        value: str,
        memory_type: str = "learned",
        importance: float = 0.5
    ) -> Dict:
        return store_memory(memory_manager, key, value, memory_type, importance)

    def _search_memories(query: str, limit: int = 5) -> Dict:
        return search_memories(memory_manager, query, limit)

    def _get_memory_summary() -> Dict:
        return get_memory_summary(memory_manager)

    def _add_goal(goal: str, priority: int = 5) -> Dict:
        return add_goal(memory_manager, goal, priority)

    def _complete_goal(goal_id: int) -> Dict:
        return complete_goal(memory_manager, goal_id)

    def _record_reflection(
        summary: str,
        content: str,
        importance: float = 0.6
    ) -> Dict:
        return record_reflection(memory_manager, summary, content, importance)

    def _get_procedures(context: str) -> Dict:
        return get_procedures(memory_manager, context)

    # Attach schema information
    _recall_memory.schema = MEMORY_TOOL_SCHEMAS["recall_memory"]
    _store_memory.schema = MEMORY_TOOL_SCHEMAS["store_memory"]
    _search_memories.schema = MEMORY_TOOL_SCHEMAS["search_memories"]
    _get_memory_summary.schema = MEMORY_TOOL_SCHEMAS["get_memory_summary"]
    _add_goal.schema = MEMORY_TOOL_SCHEMAS["add_goal"]
    _complete_goal.schema = MEMORY_TOOL_SCHEMAS["complete_goal"]
    _record_reflection.schema = MEMORY_TOOL_SCHEMAS["record_reflection"]
    _get_procedures.schema = MEMORY_TOOL_SCHEMAS["get_procedures"]

    return [
        _recall_memory,
        _store_memory,
        _search_memories,
        _get_memory_summary,
        _add_goal,
        _complete_goal,
        _record_reflection,
        _get_procedures
    ]
