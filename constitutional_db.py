"""
Constitutional Documents Database

This module creates and manages a SQLite database containing complete constitutional
documents for each country in the Constitutional Personas project.

The database schema includes:
- constitutions: Core table with metadata and full constitutional text
- amendments: Historical amendments to each constitution
- articles: Individual articles/sections for structured access
"""

import sqlite3
import os
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "constitutions.db")


@dataclass
class ConstitutionRecord:
    """A constitutional document record."""
    country: str
    official_name: str
    year_adopted: int
    year_effective: Optional[int]
    language_original: str
    full_text: str
    preamble: Optional[str]
    source_url: str
    last_amended: Optional[int]
    word_count: int
    article_count: int


def create_database():
    """Create the constitutional documents database with proper schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Main constitutions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS constitutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            country TEXT UNIQUE NOT NULL,
            official_name TEXT NOT NULL,
            persona_name TEXT,
            year_adopted INTEGER NOT NULL,
            year_effective INTEGER,
            language_original TEXT NOT NULL,
            full_text TEXT NOT NULL,
            preamble TEXT,
            source_url TEXT,
            last_amended INTEGER,
            word_count INTEGER,
            article_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Amendments table for tracking constitutional amendments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS amendments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            constitution_id INTEGER NOT NULL,
            amendment_number INTEGER,
            year_adopted INTEGER NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            summary TEXT,
            FOREIGN KEY (constitution_id) REFERENCES constitutions(id)
        )
    """)

    # Articles table for structured access to individual sections
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            constitution_id INTEGER NOT NULL,
            article_number TEXT NOT NULL,
            title TEXT,
            content TEXT NOT NULL,
            chapter TEXT,
            part TEXT,
            FOREIGN KEY (constitution_id) REFERENCES constitutions(id)
        )
    """)

    # Create indexes for efficient querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_country ON constitutions(country)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON constitutions(year_adopted)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_constitution ON articles(constitution_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_amendments_constitution ON amendments(constitution_id)")


    conn.commit()
    conn.close()
    print(f"Database created at: {DB_PATH}")


def insert_constitution(record: dict) -> int:
    """Insert a constitution record into the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO constitutions (
            country, official_name, persona_name, year_adopted, year_effective,
            language_original, full_text, preamble, source_url, last_amended,
            word_count, article_count, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record['country'],
        record['official_name'],
        record.get('persona_name'),
        record['year_adopted'],
        record.get('year_effective'),
        record['language_original'],
        record['full_text'],
        record.get('preamble'),
        record.get('source_url'),
        record.get('last_amended'),
        len(record['full_text'].split()),
        record.get('article_count', 0),
        datetime.now().isoformat()
    ))

    constitution_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return constitution_id


def get_constitution(country: str) -> Optional[dict]:
    """Retrieve a constitution by country name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM constitutions WHERE country = ?", (country,))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None


def search_constitutions(query: str) -> List[dict]:
    """Search across constitutional documents using LIKE queries."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, official_name, year_adopted, word_count
        FROM constitutions
        WHERE full_text LIKE ? OR preamble LIKE ? OR country LIKE ?
        ORDER BY year_adopted
    """, (f'%{query}%', f'%{query}%', f'%{query}%'))

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results


def list_all_constitutions() -> List[dict]:
    """List all constitutions in the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, official_name, year_adopted, last_amended, word_count
        FROM constitutions
        ORDER BY year_adopted
    """)

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results


def get_statistics() -> dict:
    """Get database statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM constitutions")
    stats['total_constitutions'] = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(word_count) FROM constitutions")
    stats['total_words'] = cursor.fetchone()[0] or 0

    cursor.execute("SELECT MIN(year_adopted), MAX(year_adopted) FROM constitutions")
    row = cursor.fetchone()
    stats['oldest_year'] = row[0]
    stats['newest_year'] = row[1]

    cursor.execute("SELECT AVG(word_count) FROM constitutions")
    stats['avg_word_count'] = int(cursor.fetchone()[0] or 0)

    conn.close()
    return stats


if __name__ == "__main__":
    create_database()
    print("Constitutional database initialized.")
