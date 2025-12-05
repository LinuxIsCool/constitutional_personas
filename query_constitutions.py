#!/usr/bin/env python3
"""
Constitutional Database Query Tool

Interactive tool for querying the constitutional documents database.

Usage:
    python query_constitutions.py                    # List all constitutions
    python query_constitutions.py --search "rights"  # Search for term
    python query_constitutions.py --country "India"  # Get specific constitution
    python query_constitutions.py --compare USA Germany  # Compare two
    python query_constitutions.py --stats            # Show statistics
"""

import sqlite3
import argparse
import re
import textwrap
from typing import Optional

DB_PATH = "constitutions.db"


def get_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def clean_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean)
    return clean.strip()


def list_all():
    """List all constitutions in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, official_name, persona_name, year_adopted,
               last_amended, word_count, language_original
        FROM constitutions
        ORDER BY year_adopted
    """)

    print("\n" + "="*80)
    print("CONSTITUTIONAL DOCUMENTS DATABASE")
    print("="*80)

    rows = cursor.fetchall()
    print(f"\nTotal: {len(rows)} constitutions\n")

    print(f"{'Country':<18} {'Persona Name':<25} {'Year':<6} {'Amended':<8} {'Words':>10}")
    print("-"*80)

    for row in rows:
        persona = row['persona_name'] or ''
        amended = str(row['last_amended']) if row['last_amended'] else 'N/A'
        print(f"{row['country']:<18} {persona[:24]:<25} {row['year_adopted']:<6} {amended:<8} {row['word_count']:>10,}")

    conn.close()


def search(query: str):
    """Search constitutional texts for a term."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, year_adopted, full_text
        FROM constitutions
        WHERE full_text LIKE ?
        ORDER BY year_adopted
    """, (f'%{query}%',))

    rows = cursor.fetchall()

    print(f"\n{'='*70}")
    print(f"SEARCH RESULTS: '{query}'")
    print(f"{'='*70}")
    print(f"Found in {len(rows)} constitutions:\n")

    for row in rows:
        text = clean_html(row['full_text'])
        # Find and extract context around the search term
        idx = text.lower().find(query.lower())
        if idx >= 0:
            start = max(0, idx - 50)
            end = min(len(text), idx + len(query) + 100)
            excerpt = text[start:end]
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(text):
                excerpt = excerpt + "..."

            print(f"• {row['country']} ({row['year_adopted']})")
            print(f"  \"{excerpt}\"")
            print()

    conn.close()


def get_country(country: str):
    """Get full constitution for a specific country."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM constitutions
        WHERE country LIKE ?
    """, (f'%{country}%',))

    row = cursor.fetchone()

    if not row:
        print(f"No constitution found for: {country}")
        conn.close()
        return

    text = clean_html(row['full_text'])

    print(f"\n{'='*80}")
    print(f"{row['official_name']}")
    print(f"{'='*80}")
    print(f"Country:        {row['country']}")
    print(f"Persona:        {row['persona_name']}")
    print(f"Year Adopted:   {row['year_adopted']}")
    print(f"Year Effective: {row['year_effective'] or 'Same year'}")
    print(f"Last Amended:   {row['last_amended'] or 'N/A'}")
    print(f"Original Lang:  {row['language_original']}")
    print(f"Word Count:     {row['word_count']:,}")
    print(f"Source:         {row['source_url']}")
    print(f"\n{'-'*80}")
    print("FULL TEXT (first 3000 characters):")
    print("-"*80)
    print(textwrap.fill(text[:3000], width=80))
    if len(text) > 3000:
        print(f"\n... [{len(text) - 3000:,} more characters]")

    conn.close()


def compare_countries(country1: str, country2: str):
    """Compare two constitutions."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, official_name, persona_name, year_adopted,
               last_amended, word_count, language_original
        FROM constitutions
        WHERE country LIKE ? OR country LIKE ?
    """, (f'%{country1}%', f'%{country2}%'))

    rows = cursor.fetchall()

    if len(rows) < 2:
        print("Could not find both countries")
        conn.close()
        return

    print(f"\n{'='*80}")
    print("CONSTITUTION COMPARISON")
    print(f"{'='*80}\n")

    print(f"{'Attribute':<20} {rows[0]['country']:<28} {rows[1]['country']:<28}")
    print("-"*80)
    print(f"{'Persona':<20} {(rows[0]['persona_name'] or 'N/A'):<28} {(rows[1]['persona_name'] or 'N/A'):<28}")
    print(f"{'Year Adopted':<20} {rows[0]['year_adopted']:<28} {rows[1]['year_adopted']:<28}")
    print(f"{'Last Amended':<20} {str(rows[0]['last_amended'] or 'N/A'):<28} {str(rows[1]['last_amended'] or 'N/A'):<28}")
    print(f"{'Word Count':<20} {rows[0]['word_count']:,<27} {rows[1]['word_count']:,<27}")
    print(f"{'Original Language':<20} {rows[0]['language_original']:<28} {rows[1]['language_original']:<28}")

    # Calculate age difference
    age_diff = abs(rows[0]['year_adopted'] - rows[1]['year_adopted'])
    print(f"\nAge difference: {age_diff} years")

    conn.close()


def show_stats():
    """Show database statistics."""
    conn = get_connection()
    cursor = conn.cursor()

    print(f"\n{'='*70}")
    print("DATABASE STATISTICS")
    print(f"{'='*70}\n")

    # Basic stats
    cursor.execute("SELECT COUNT(*) FROM constitutions")
    total = cursor.fetchone()[0]
    print(f"Total constitutions: {total}")

    cursor.execute("SELECT SUM(word_count) FROM constitutions")
    total_words = cursor.fetchone()[0]
    print(f"Total words:         {total_words:,}")

    cursor.execute("SELECT AVG(word_count) FROM constitutions")
    avg_words = cursor.fetchone()[0]
    print(f"Average word count:  {int(avg_words):,}")

    # Longest and shortest
    cursor.execute("""
        SELECT country, word_count FROM constitutions
        ORDER BY word_count DESC LIMIT 1
    """)
    longest = cursor.fetchone()
    print(f"Longest:             {longest['country']} ({longest['word_count']:,} words)")

    cursor.execute("""
        SELECT country, word_count FROM constitutions
        ORDER BY word_count ASC LIMIT 1
    """)
    shortest = cursor.fetchone()
    print(f"Shortest:            {shortest['country']} ({shortest['word_count']:,} words)")

    # Oldest and newest
    cursor.execute("""
        SELECT country, year_adopted FROM constitutions
        ORDER BY year_adopted ASC LIMIT 1
    """)
    oldest = cursor.fetchone()
    print(f"Oldest:              {oldest['country']} ({oldest['year_adopted']})")

    cursor.execute("""
        SELECT country, year_adopted FROM constitutions
        ORDER BY year_adopted DESC LIMIT 1
    """)
    newest = cursor.fetchone()
    print(f"Newest:              {newest['country']} ({newest['year_adopted']})")

    # By decade
    print(f"\n{'='*70}")
    print("CONSTITUTIONS BY DECADE")
    print(f"{'='*70}\n")

    cursor.execute("""
        SELECT (year_adopted / 10) * 10 as decade, COUNT(*) as count, GROUP_CONCAT(country) as countries
        FROM constitutions
        GROUP BY decade
        ORDER BY decade
    """)

    for row in cursor.fetchall():
        countries = row['countries'].split(',')
        print(f"{row['decade']}s: {row['count']} constitution(s)")
        for c in countries:
            print(f"       • {c}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Query the Constitutional Documents Database')
    parser.add_argument('--search', '-s', type=str, help='Search for a term in all constitutions')
    parser.add_argument('--country', '-c', type=str, help='Get full details for a specific country')
    parser.add_argument('--compare', nargs=2, metavar=('COUNTRY1', 'COUNTRY2'), help='Compare two constitutions')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')

    args = parser.parse_args()

    if args.search:
        search(args.search)
    elif args.country:
        get_country(args.country)
    elif args.compare:
        compare_countries(args.compare[0], args.compare[1])
    elif args.stats:
        show_stats()
    else:
        list_all()


if __name__ == "__main__":
    main()
