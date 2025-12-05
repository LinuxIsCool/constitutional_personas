"""
Populate Constitutional Documents Database

This script fetches complete constitutional texts from the Constitute Project
and other authoritative sources, then populates the SQLite database.
"""

import urllib.request
import urllib.error
import json
import re
import ssl
import time
from typing import Optional
from constitutional_db import create_database, insert_constitution, DB_PATH, get_statistics, list_all_constitutions

# Constitutional metadata for all countries in the project
CONSTITUTIONAL_METADATA = {
    "United States": {
        "official_name": "Constitution of the United States of America",
        "persona_name": "The Founders' Covenant",
        "year_adopted": 1787,
        "year_effective": 1789,
        "language_original": "English",
        "last_amended": 1992,
        "source_url": "https://www.constituteproject.org/constitution/United_States_of_America_1992",
        "constitute_id": "United_States_of_America_1992"
    },
    "Germany": {
        "official_name": "Grundgesetz für die Bundesrepublik Deutschland (Basic Law for the Federal Republic of Germany)",
        "persona_name": "The Phoenix Charter",
        "year_adopted": 1949,
        "year_effective": 1949,
        "language_original": "German",
        "last_amended": 2014,
        "source_url": "https://www.constituteproject.org/constitution/German_Federal_Republic_2014",
        "constitute_id": "German_Federal_Republic_2014"
    },
    "France": {
        "official_name": "Constitution de la République française (Constitution of the French Republic)",
        "persona_name": "The Republic's Voice",
        "year_adopted": 1958,
        "year_effective": 1958,
        "language_original": "French",
        "last_amended": 2008,
        "source_url": "https://www.constituteproject.org/constitution/France_2008",
        "constitute_id": "France_2008"
    },
    "India": {
        "official_name": "Constitution of India",
        "persona_name": "The Mosaic Compact",
        "year_adopted": 1949,
        "year_effective": 1950,
        "language_original": "English/Hindi",
        "last_amended": 2016,
        "source_url": "https://www.constituteproject.org/constitution/India_2016",
        "constitute_id": "India_2016"
    },
    "South Africa": {
        "official_name": "Constitution of the Republic of South Africa",
        "persona_name": "The Rainbow Covenant",
        "year_adopted": 1996,
        "year_effective": 1997,
        "language_original": "English",
        "last_amended": 2012,
        "source_url": "https://www.constituteproject.org/constitution/South_Africa_2012",
        "constitute_id": "South_Africa_2012"
    },
    "Japan": {
        "official_name": "Nihon-koku Kenpō (Constitution of Japan)",
        "persona_name": "The Pacifist's Oath",
        "year_adopted": 1946,
        "year_effective": 1947,
        "language_original": "Japanese",
        "last_amended": 1946,
        "source_url": "https://www.constituteproject.org/constitution/Japan_1946",
        "constitute_id": "Japan_1946"
    },
    "Sweden": {
        "official_name": "Sveriges grundlagar (Fundamental Laws of Sweden)",
        "persona_name": "The Social Pact",
        "year_adopted": 1974,
        "year_effective": 1975,
        "language_original": "Swedish",
        "last_amended": 2012,
        "source_url": "https://www.constituteproject.org/constitution/Sweden_2012",
        "constitute_id": "Sweden_2012"
    },
    "Switzerland": {
        "official_name": "Bundesverfassung der Schweizerischen Eidgenossenschaft (Federal Constitution of the Swiss Confederation)",
        "persona_name": "The Alpine Concordat",
        "year_adopted": 1999,
        "year_effective": 2000,
        "language_original": "German/French/Italian/Romansh",
        "last_amended": 2014,
        "source_url": "https://www.constituteproject.org/constitution/Switzerland_2014",
        "constitute_id": "Switzerland_2014"
    },
    "Costa Rica": {
        "official_name": "Constitución Política de la República de Costa Rica",
        "persona_name": "The Verdant Charter",
        "year_adopted": 1949,
        "year_effective": 1949,
        "language_original": "Spanish",
        "last_amended": 2020,
        "source_url": "https://www.constituteproject.org/constitution/Costa_Rica_2020",
        "constitute_id": "Costa_Rica_2020"
    },
    "Canada": {
        "official_name": "Constitution Act / Loi constitutionnelle",
        "persona_name": "The Maple Accord",
        "year_adopted": 1867,
        "year_effective": 1867,
        "language_original": "English/French",
        "last_amended": 2011,
        "source_url": "https://www.constituteproject.org/constitution/Canada_2011",
        "constitute_id": "Canada_2011"
    },
    "Estonia": {
        "official_name": "Eesti Vabariigi põhiseadus (Constitution of the Republic of Estonia)",
        "persona_name": "The Digital Republic",
        "year_adopted": 1992,
        "year_effective": 1992,
        "language_original": "Estonian",
        "last_amended": 2015,
        "source_url": "https://www.constituteproject.org/constitution/Estonia_2015",
        "constitute_id": "Estonia_2015"
    },
    "Brazil": {
        "official_name": "Constituição da República Federativa do Brasil",
        "persona_name": "The Amazonian Covenant",
        "year_adopted": 1988,
        "year_effective": 1988,
        "language_original": "Portuguese",
        "last_amended": 2017,
        "source_url": "https://www.constituteproject.org/constitution/Brazil_2017",
        "constitute_id": "Brazil_2017"
    },
    "Norway": {
        "official_name": "Kongeriket Norges Grunnlov (Constitution of the Kingdom of Norway)",
        "persona_name": "The Fjord Charter",
        "year_adopted": 1814,
        "year_effective": 1814,
        "language_original": "Norwegian",
        "last_amended": 2016,
        "source_url": "https://www.constituteproject.org/constitution/Norway_2016",
        "constitute_id": "Norway_2016"
    },
    "South Korea": {
        "official_name": "Daehanminguk Heonbeop (Constitution of the Republic of Korea)",
        "persona_name": "The Morning Calm",
        "year_adopted": 1948,
        "year_effective": 1948,
        "language_original": "Korean",
        "last_amended": 1987,
        "source_url": "https://www.constituteproject.org/constitution/Republic_of_Korea_1987",
        "constitute_id": "Republic_of_Korea_1987"
    },
    "Ireland": {
        "official_name": "Bunreacht na hÉireann (Constitution of Ireland)",
        "persona_name": "The Emerald Covenant",
        "year_adopted": 1937,
        "year_effective": 1937,
        "language_original": "Irish/English",
        "last_amended": 2019,
        "source_url": "https://www.constituteproject.org/constitution/Ireland_2019",
        "constitute_id": "Ireland_2019"
    },
    "Australia": {
        "official_name": "Commonwealth of Australia Constitution Act",
        "persona_name": "The Southern Cross",
        "year_adopted": 1900,
        "year_effective": 1901,
        "language_original": "English",
        "last_amended": 1985,
        "source_url": "https://www.constituteproject.org/constitution/Australia_1985",
        "constitute_id": "Australia_1985"
    },
    "New Zealand": {
        "official_name": "Constitution of New Zealand (Uncodified)",
        "persona_name": "The Tui's Song",
        "year_adopted": 1852,
        "year_effective": 1852,
        "language_original": "English",
        "last_amended": 2014,
        "source_url": "https://www.constituteproject.org/constitution/New_Zealand_2014",
        "constitute_id": "New_Zealand_2014"
    },
    "Spain": {
        "official_name": "Constitución Española (Spanish Constitution)",
        "persona_name": "The Iberian Spring",
        "year_adopted": 1978,
        "year_effective": 1978,
        "language_original": "Spanish",
        "last_amended": 2011,
        "source_url": "https://www.constituteproject.org/constitution/Spain_2011",
        "constitute_id": "Spain_2011"
    }
}


def fetch_constitution_text(constitute_id: str) -> Optional[str]:
    """
    Fetch constitutional text from the Constitute Project API.

    The Constitute Project provides constitutional texts in a structured format.
    """
    # Use their JSON API endpoint
    api_url = f"https://www.constituteproject.org/constitution/{constitute_id}?lang=en"

    try:
        # Create SSL context that doesn't verify (for compatibility)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }

        req = urllib.request.Request(api_url, headers=headers)

        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            html = response.read().decode('utf-8')

            # Extract the constitutional text from the page
            # The text is embedded in a data structure or the HTML body
            return html

    except urllib.error.HTTPError as e:
        print(f"HTTP Error fetching {constitute_id}: {e.code}")
        return None
    except urllib.error.URLError as e:
        print(f"URL Error fetching {constitute_id}: {e.reason}")
        return None
    except Exception as e:
        print(f"Error fetching {constitute_id}: {e}")
        return None


def extract_preamble(text: str) -> Optional[str]:
    """Extract the preamble from constitutional text."""
    # Common patterns for preambles
    preamble_patterns = [
        r'(?i)PREAMBLE\s*\n(.*?)(?=\n\s*(?:ARTICLE|CHAPTER|PART|SECTION)\s*[IVX1-9])',
        r'(?i)^We[,\s]the\s+[Pp]eople.*?(?=\n\s*(?:ARTICLE|CHAPTER|PART|SECTION))',
    ]

    for pattern in preamble_patterns:
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip() if match.lastindex else match.group(0).strip()

    return None


def count_articles(text: str) -> int:
    """Count the number of articles in constitutional text."""
    # Match various article numbering patterns
    patterns = [
        r'(?i)\bARTICLE\s+\d+',
        r'(?i)\bArt\.\s*\d+',
        r'(?i)\bSection\s+\d+',
        r'§\s*\d+',
    ]

    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text)
        count = max(count, len(matches))

    return count


def populate_database():
    """Populate the database with all constitutional documents."""
    print("=" * 70)
    print("CONSTITUTIONAL DOCUMENTS DATABASE POPULATION")
    print("=" * 70)
    print()

    # Create database if it doesn't exist
    create_database()

    successful = 0
    failed = []

    for country, metadata in CONSTITUTIONAL_METADATA.items():
        print(f"\nProcessing: {country}...")

        # Fetch the constitutional text
        raw_html = fetch_constitution_text(metadata['constitute_id'])

        if raw_html:
            # For now, store the raw content
            # In production, we'd parse the HTML to extract clean text
            full_text = raw_html

            # Extract preamble
            preamble = extract_preamble(full_text)

            # Count articles
            article_count = count_articles(full_text)

            # Prepare record
            record = {
                'country': country,
                'official_name': metadata['official_name'],
                'persona_name': metadata['persona_name'],
                'year_adopted': metadata['year_adopted'],
                'year_effective': metadata.get('year_effective'),
                'language_original': metadata['language_original'],
                'full_text': full_text,
                'preamble': preamble,
                'source_url': metadata['source_url'],
                'last_amended': metadata.get('last_amended'),
                'article_count': article_count
            }

            # Insert into database
            constitution_id = insert_constitution(record)
            print(f"  ✓ Inserted: {country} (ID: {constitution_id})")
            print(f"    Words: {len(full_text.split()):,}")
            successful += 1
        else:
            print(f"  ✗ Failed to fetch: {country}")
            failed.append(country)

        # Rate limiting
        time.sleep(1)

    print()
    print("=" * 70)
    print("POPULATION COMPLETE")
    print("=" * 70)
    print(f"Successful: {successful}/{len(CONSTITUTIONAL_METADATA)}")

    if failed:
        print(f"Failed: {', '.join(failed)}")

    # Print statistics
    stats = get_statistics()
    print()
    print("Database Statistics:")
    print(f"  Total constitutions: {stats['total_constitutions']}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Average word count: {stats['avg_word_count']:,}")
    print(f"  Year range: {stats['oldest_year']} - {stats['newest_year']}")


def list_database_contents():
    """List all constitutions in the database."""
    print("\n" + "=" * 70)
    print("CONSTITUTIONAL DATABASE CONTENTS")
    print("=" * 70)

    constitutions = list_all_constitutions()

    if not constitutions:
        print("Database is empty.")
        return

    print(f"\n{'Country':<20} {'Official Name':<40} {'Year':<6} {'Words':>10}")
    print("-" * 80)

    for c in constitutions:
        name = c['official_name'][:37] + "..." if len(c['official_name']) > 40 else c['official_name']
        words = f"{c['word_count']:,}" if c['word_count'] else "N/A"
        print(f"{c['country']:<20} {name:<40} {c['year_adopted']:<6} {words:>10}")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_database_contents()
    else:
        populate_database()
        list_database_contents()
