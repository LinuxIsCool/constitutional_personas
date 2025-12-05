"""
Database Module for Constitutional Personas

Contains database schemas and utilities for:
- Constitutional documents database
- Agent memory database (5 memory types)
"""

from .memory_schema import create_memory_database, get_memory_connection, MEMORY_DB_PATH

__all__ = [
    'create_memory_database',
    'get_memory_connection',
    'MEMORY_DB_PATH'
]
