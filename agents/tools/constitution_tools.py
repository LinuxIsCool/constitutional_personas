"""
Constitution Tools for Constitutional Agents

These tools enable agents to search and retrieve their constitutional text
using RAG (Retrieval Augmented Generation).

Usage with Claude Agent SDK:
    from claude_agent_sdk import tool, create_sdk_mcp_server
    from agents.tools.constitution_tools import create_constitution_tools

    tools = create_constitution_tools(semantic_memory)
    server = create_sdk_mcp_server(name="constitution", tools=tools)
"""

import json
from typing import Dict, Any, List, Optional, Callable

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memory.semantic_memory import SemanticMemory


def create_tool_response(content: str, is_error: bool = False) -> Dict:
    """Create a standard tool response."""
    return {
        "content": [{"type": "text", "text": content}],
        "is_error": is_error
    }


def search_constitution(
    semantic: SemanticMemory,
    query: str,
    top_k: int = 5
) -> Dict:
    """
    Search the constitution for relevant passages.

    Args:
        semantic: The SemanticMemory instance
        query: Search query
        top_k: Number of results

    Returns:
        Tool response with relevant passages
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    results = semantic.retrieve(query, top_k=top_k, min_similarity=0.2)

    if not results:
        return create_tool_response(f"No relevant passages found for: {query}")

    formatted = []
    for r in results:
        section = r.section_title or r.content_type.title()
        formatted.append({
            "section": section,
            "content": r.content,
            "relevance": f"{r.similarity:.1%}"
        })

    return create_tool_response(json.dumps(formatted, indent=2))


def get_article(
    semantic: SemanticMemory,
    article_identifier: str
) -> Dict:
    """
    Get a specific article or section from the constitution.

    Args:
        semantic: The SemanticMemory instance
        article_identifier: Article number or name (e.g., "Article 1", "First Amendment")

    Returns:
        Tool response with article content
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    # Search for the specific article
    results = semantic.retrieve(
        query=article_identifier,
        top_k=5,
        content_types=['article', 'amendment', 'section'],
        min_similarity=0.2
    )

    if not results:
        return create_tool_response(f"No article found matching: {article_identifier}")

    # Filter for best matches
    formatted = []
    for r in results:
        if r.section_title and article_identifier.lower() in r.section_title.lower():
            formatted.append({
                "section": r.section_title,
                "number": r.section_number,
                "content": r.content
            })

    # If no exact match, return best semantic matches
    if not formatted:
        formatted = [
            {
                "section": r.section_title or r.content_type,
                "content": r.content,
                "relevance": f"{r.similarity:.1%}"
            }
            for r in results[:3]
        ]

    return create_tool_response(json.dumps(formatted, indent=2))


def get_preamble(semantic: SemanticMemory) -> Dict:
    """
    Get the constitution's preamble.

    Args:
        semantic: The SemanticMemory instance

    Returns:
        Tool response with preamble content
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    results = semantic.retrieve_preamble()

    if not results:
        # Try searching for preamble content
        results = semantic.retrieve("preamble founding principles", top_k=3)

    if not results:
        return create_tool_response("No preamble found in this constitution.")

    # Combine preamble chunks
    preamble_text = "\n\n".join(r.content for r in results)

    return create_tool_response(preamble_text)


def get_rights(
    semantic: SemanticMemory,
    right_type: Optional[str] = None
) -> Dict:
    """
    Get constitutional rights provisions.

    Args:
        semantic: The SemanticMemory instance
        right_type: Optional specific right (e.g., "speech", "religion", "privacy")

    Returns:
        Tool response with rights provisions
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    query = "rights freedoms liberties"
    if right_type:
        query = f"{right_type} right freedom liberty"

    results = semantic.retrieve(query, top_k=5, min_similarity=0.25)

    if not results:
        return create_tool_response(f"No rights provisions found matching: {right_type or 'general'}")

    formatted = [
        {
            "section": r.section_title or "Rights Provision",
            "content": r.content,
            "relevance": f"{r.similarity:.1%}"
        }
        for r in results
    ]

    return create_tool_response(json.dumps(formatted, indent=2))


def get_structure(
    semantic: SemanticMemory,
    branch: Optional[str] = None
) -> Dict:
    """
    Get constitutional provisions about government structure.

    Args:
        semantic: The SemanticMemory instance
        branch: Optional specific branch (executive, legislative, judicial)

    Returns:
        Tool response with structure provisions
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    if branch:
        query = f"{branch} branch powers duties"
    else:
        query = "government structure branches powers separation"

    results = semantic.retrieve(query, top_k=5, min_similarity=0.25)

    if not results:
        return create_tool_response(f"No structural provisions found for: {branch or 'general'}")

    formatted = [
        {
            "section": r.section_title or "Structure",
            "content": r.content,
            "relevance": f"{r.similarity:.1%}"
        }
        for r in results
    ]

    return create_tool_response(json.dumps(formatted, indent=2))


def compare_provisions(
    semantic: SemanticMemory,
    topic: str
) -> Dict:
    """
    Get constitutional provisions on a topic for comparison purposes.

    Args:
        semantic: The SemanticMemory instance
        topic: The topic to find provisions about

    Returns:
        Tool response with provisions for comparison
    """
    if not semantic.is_initialized():
        return create_tool_response(
            "Constitutional text not yet embedded. Please initialize semantic memory first.",
            is_error=True
        )

    results = semantic.retrieve(topic, top_k=5, min_similarity=0.2)

    if not results:
        return create_tool_response(f"No provisions found for topic: {topic}")

    formatted = {
        "topic": topic,
        "constitution": semantic.agent_id,
        "provisions": [
            {
                "section": r.section_title or r.content_type,
                "content": r.content
            }
            for r in results
        ]
    }

    return create_tool_response(json.dumps(formatted, indent=2))


# Tool schema definitions
CONSTITUTION_TOOL_SCHEMAS = {
    "search_constitution": {
        "name": "search_constitution",
        "description": "Search the constitution for passages relevant to a query using semantic similarity",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or topic"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    "get_article": {
        "name": "get_article",
        "description": "Get a specific article, section, or amendment from the constitution",
        "input_schema": {
            "type": "object",
            "properties": {
                "article_identifier": {
                    "type": "string",
                    "description": "Article identifier (e.g., 'Article 1', 'First Amendment', 'Section 2')"
                }
            },
            "required": ["article_identifier"]
        }
    },
    "get_preamble": {
        "name": "get_preamble",
        "description": "Get the constitution's preamble",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    "get_rights": {
        "name": "get_rights",
        "description": "Get constitutional provisions about rights and freedoms",
        "input_schema": {
            "type": "object",
            "properties": {
                "right_type": {
                    "type": "string",
                    "description": "Optional specific right type (e.g., 'speech', 'religion', 'privacy')"
                }
            }
        }
    },
    "get_structure": {
        "name": "get_structure",
        "description": "Get constitutional provisions about government structure",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "enum": ["executive", "legislative", "judicial"],
                    "description": "Optional specific government branch"
                }
            }
        }
    },
    "compare_provisions": {
        "name": "compare_provisions",
        "description": "Get constitutional provisions on a topic for comparison with other constitutions",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to find provisions about"
                }
            },
            "required": ["topic"]
        }
    }
}


def create_constitution_tools(semantic_memory: SemanticMemory) -> List[Callable]:
    """
    Create constitution tools bound to a specific semantic memory.

    Args:
        semantic_memory: The SemanticMemory instance to bind to

    Returns:
        List of tool functions ready for SDK registration
    """

    def _search_constitution(query: str, top_k: int = 5) -> Dict:
        return search_constitution(semantic_memory, query, top_k)

    def _get_article(article_identifier: str) -> Dict:
        return get_article(semantic_memory, article_identifier)

    def _get_preamble() -> Dict:
        return get_preamble(semantic_memory)

    def _get_rights(right_type: Optional[str] = None) -> Dict:
        return get_rights(semantic_memory, right_type)

    def _get_structure(branch: Optional[str] = None) -> Dict:
        return get_structure(semantic_memory, branch)

    def _compare_provisions(topic: str) -> Dict:
        return compare_provisions(semantic_memory, topic)

    # Attach schema information
    _search_constitution.schema = CONSTITUTION_TOOL_SCHEMAS["search_constitution"]
    _get_article.schema = CONSTITUTION_TOOL_SCHEMAS["get_article"]
    _get_preamble.schema = CONSTITUTION_TOOL_SCHEMAS["get_preamble"]
    _get_rights.schema = CONSTITUTION_TOOL_SCHEMAS["get_rights"]
    _get_structure.schema = CONSTITUTION_TOOL_SCHEMAS["get_structure"]
    _compare_provisions.schema = CONSTITUTION_TOOL_SCHEMAS["compare_provisions"]

    return [
        _search_constitution,
        _get_article,
        _get_preamble,
        _get_rights,
        _get_structure,
        _compare_provisions
    ]
