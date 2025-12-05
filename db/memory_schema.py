"""
Memory Database Schema for Constitutional Agents

This module defines the database schema for the five-memory cognitive architecture:
1. Persistent Memory - Long-term facts, beliefs, preferences (survives sessions)
2. Working Memory - Current context, active goals, scratchpad (session-scoped)
3. Episodic Memory - Timestamped experiences and interactions
4. Procedural Memory - Behavioral patterns and response strategies
5. Semantic Memory - Embedded constitutional chunks for RAG retrieval
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional

MEMORY_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory.db")


def create_memory_database(db_path: str = MEMORY_DB_PATH):
    """Create the complete memory database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # =========================================================================
    # PERSISTENT MEMORY
    # Long-term storage of facts, beliefs, preferences, and learned information
    # Survives across all sessions
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persistent_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            memory_key TEXT NOT NULL,
            memory_value TEXT NOT NULL,
            memory_type TEXT DEFAULT 'fact',  -- fact, belief, preference, learned
            importance REAL DEFAULT 0.5,       -- 0.0 to 1.0
            confidence REAL DEFAULT 0.8,       -- 0.0 to 1.0
            source TEXT,                       -- where this memory came from
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            last_accessed TIMESTAMP,
            metadata TEXT,  -- JSON for additional attributes
            UNIQUE(agent_id, memory_key)
        )
    """)

    # =========================================================================
    # WORKING MEMORY
    # Current session context, active goals, temporary information
    # Cleared or archived between sessions
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,  -- goal, context, scratchpad, focus
            content TEXT NOT NULL,
            priority INTEGER DEFAULT 5,  -- 1-10, higher = more important
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,         -- optional expiration
            is_active BOOLEAN DEFAULT 1,
            parent_id INTEGER,            -- for hierarchical goals
            metadata TEXT,
            FOREIGN KEY (parent_id) REFERENCES working_memory(id)
        )
    """)

    # =========================================================================
    # EPISODIC MEMORY
    # Specific experiences, interactions, and events with temporal context
    # "I remember when..."
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            episode_type TEXT NOT NULL,   -- conversation, query, reflection, event
            summary TEXT NOT NULL,         -- brief description of episode
            content TEXT NOT NULL,         -- full episode content
            participants TEXT,             -- JSON list of participants
            emotional_valence REAL,        -- -1.0 (negative) to 1.0 (positive)
            importance REAL DEFAULT 0.5,   -- 0.0 to 1.0
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            duration_seconds INTEGER,
            location TEXT,                 -- context location (topic/domain)
            tags TEXT,                     -- JSON list of tags
            linked_episodes TEXT,          -- JSON list of related episode IDs
            embedding BLOB,                -- vector embedding for similarity search
            metadata TEXT
        )
    """)

    # =========================================================================
    # PROCEDURAL MEMORY
    # How to do things - behavioral patterns, response strategies, routines
    # "When X happens, I typically do Y"
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS procedural_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            procedure_name TEXT NOT NULL,
            trigger_pattern TEXT NOT NULL,  -- what activates this procedure
            action_sequence TEXT NOT NULL,  -- JSON list of steps/actions
            preconditions TEXT,             -- JSON conditions that must be true
            postconditions TEXT,            -- JSON expected outcomes
            success_rate REAL DEFAULT 0.5,  -- historical success rate
            usage_count INTEGER DEFAULT 0,
            last_used TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT,                    -- constitutional, learned, default
            priority INTEGER DEFAULT 5,     -- conflict resolution priority
            is_active BOOLEAN DEFAULT 1,
            metadata TEXT,
            UNIQUE(agent_id, procedure_name)
        )
    """)

    # =========================================================================
    # SEMANTIC MEMORY
    # Factual knowledge, concepts, and constitutional chunks with embeddings
    # Supports RAG retrieval
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS semantic_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,         -- unique identifier for chunk
            content TEXT NOT NULL,          -- the actual text content
            content_type TEXT NOT NULL,     -- constitution, article, amendment, fact
            source_document TEXT,           -- which document this came from
            section_title TEXT,             -- article/section title
            section_number TEXT,            -- article/section number
            start_position INTEGER,         -- character position in source
            end_position INTEGER,
            embedding BLOB NOT NULL,        -- vector embedding
            embedding_model TEXT NOT NULL,  -- model used for embedding
            token_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,                  -- JSON for additional attributes
            UNIQUE(agent_id, chunk_id)
        )
    """)

    # =========================================================================
    # AGENT SESSIONS
    # Track agent sessions for working memory management
    # =========================================================================
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

    # =========================================================================
    # MEMORY ASSOCIATIONS
    # Links between different memory types for associative retrieval
    # =========================================================================
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_associations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT NOT NULL,
            source_type TEXT NOT NULL,      -- persistent, episodic, procedural, semantic
            source_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            target_id INTEGER NOT NULL,
            association_type TEXT NOT NULL, -- causal, temporal, semantic, hierarchical
            strength REAL DEFAULT 0.5,      -- 0.0 to 1.0
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT
        )
    """)

    # =========================================================================
    # INDEXES
    # =========================================================================
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persistent_agent ON persistent_memory(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_persistent_key ON persistent_memory(agent_id, memory_key)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_working_session ON working_memory(agent_id, session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_working_active ON working_memory(agent_id, is_active)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_agent ON episodic_memory(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_time ON episodic_memory(agent_id, timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic_memory(agent_id, episode_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedural_agent ON procedural_memory(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_procedural_trigger ON procedural_memory(agent_id, trigger_pattern)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memory(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_type ON semantic_memory(agent_id, content_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_agent ON agent_sessions(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_associations_source ON memory_associations(agent_id, source_type, source_id)")

    conn.commit()
    conn.close()
    print(f"Memory database created at: {db_path}")


def get_memory_connection(db_path: str = MEMORY_DB_PATH) -> sqlite3.Connection:
    """Get a connection to the memory database with row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


if __name__ == "__main__":
    create_memory_database()
    print("Memory schema initialized successfully.")
