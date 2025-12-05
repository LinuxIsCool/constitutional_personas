"""
Constitution Chunker for RAG

Intelligently chunks constitutional documents for embedding and retrieval.
Uses multiple strategies:
1. Article-based chunking - respects document structure
2. Semantic chunking - breaks on sentence/paragraph boundaries
3. Overlapping windows - ensures context continuity
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Generator
import html
import hashlib


@dataclass
class Chunk:
    """A chunk of constitutional text with metadata."""
    chunk_id: str
    content: str
    content_type: str  # preamble, article, section, amendment, paragraph
    source_document: str
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    start_position: int = 0
    end_position: int = 0
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)


class ConstitutionChunker:
    """
    Intelligent chunker for constitutional documents.

    Strategies:
    - article: Chunk by articles/sections (respects document structure)
    - paragraph: Chunk by paragraphs with size limits
    - sliding: Sliding window with overlap
    - hybrid: Combine article-aware with size limits
    """

    def __init__(
        self,
        max_chunk_size: int = 500,  # target tokens per chunk
        min_chunk_size: int = 100,  # minimum tokens
        overlap_size: int = 50,     # overlap tokens for sliding window
        strategy: str = "hybrid"
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.strategy = strategy

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags, JS templates, and decode entities."""
        # Remove script and style elements
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<noscript[^>]*>.*?</noscript>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<head[^>]*>.*?</head>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Remove Angular/JS template expressions
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'\{\{.*?\}\}', '', text)
        text = re.sub(r'ng-[a-z-]+="[^"]*"', '', text)
        text = re.sub(r'data-ng-[a-z-]+="[^"]*"', '', text)

        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Decode HTML entities
        text = html.unescape(text)

        # Remove common boilerplate patterns
        text = re.sub(r'Log in.*?Log out', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'Constitutions\s*Countries\s*Topics', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Source of constitutional authority.*?Preamble', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)

        return text.strip()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 chars per token for English)."""
        return len(text) // 4

    def _generate_chunk_id(self, country: str, content: str, index: int) -> str:
        """Generate a unique chunk ID."""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{country.lower().replace(' ', '_')}_{index:04d}_{content_hash}"

    def _extract_articles(self, text: str) -> List[dict]:
        """Extract articles/sections from constitutional text."""
        articles = []

        # Common article/section patterns in constitutions
        # Each tuple: (pattern, type_group, num_group, title_group)
        patterns = [
            # "Article 1" or "Article I"
            (r'(Article)\s+([\dIVXLCDM]+)[.\s:]+([^\n]*)', 1, 2, 3),
            # "Section 1"
            (r'(Section)\s+(\d+)[.\s:]*([^\n]*)', 1, 2, 3),
            # "Chapter 1"
            (r'(Chapter)\s+([\dIVXLCDM]+)[.\s:]+([^\n]*)', 1, 2, 3),
            # "Part I"
            (r'(Part)\s+([\dIVXLCDM]+)[.\s:]+([^\n]*)', 1, 2, 3),
            # "Amendment 1" or "Amendment I"
            (r'(Amendment)\s+([\dIVXLCDM]+)[.\s:]*([^\n]*)', 1, 2, 3),
            # German style "Artikel 1"
            (r'(Artikel)\s+(\d+)[.\s:]*([^\n]*)', 1, 2, 3),
        ]

        # Process each pattern separately to avoid named group conflicts
        for pattern, type_idx, num_idx, title_idx in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                article_type = match.group(type_idx)
                article_num = match.group(num_idx)
                article_title = match.group(title_idx).strip() if match.group(title_idx) else ''

                articles.append({
                    'type': article_type,
                    'number': article_num,
                    'title': article_title,
                    'start': match.start(),
                    'end': match.end()
                })

        # Sort by position and remove duplicates
        articles.sort(key=lambda x: x['start'])

        # Remove duplicates (overlapping matches)
        unique_articles = []
        last_end = -1
        for article in articles:
            if article['start'] >= last_end:
                unique_articles.append(article)
                last_end = article['end']

        return unique_articles

    def _extract_preamble(self, text: str) -> Optional[dict]:
        """Extract preamble from constitutional text."""
        # Common preamble patterns
        patterns = [
            r'(?i)^.*?PREAMBLE\s*\n(.*?)(?=\n\s*(?:Article|Chapter|Part|Section)\s*[IVX1-9])',
            r'(?i)^(We[,\s]+the\s+[Pp]eople.*?)(?=\n\s*(?:Article|Chapter|Part|Section))',
            r'(?i)^(IN THE NAME OF.*?)(?=\n\s*(?:Article|Chapter|Part|Section))',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
            if match:
                content = match.group(1).strip()
                if len(content) > 50:  # Ensure it's substantial
                    return {
                        'type': 'preamble',
                        'content': content,
                        'start': match.start(1),
                        'end': match.end(1)
                    }

        return None

    def _chunk_by_sentences(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks by sentence boundaries."""
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = self._estimate_tokens(sentence)

            if current_size + sentence_size > max_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _sliding_window_chunks(self, text: str) -> List[tuple]:
        """Create overlapping chunks using sliding window."""
        words = text.split()
        chunks = []
        word_chunk_size = self.max_chunk_size  # Approximate words per chunk
        overlap_words = self.overlap_size

        start = 0
        while start < len(words):
            end = min(start + word_chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Calculate approximate character positions
            char_start = len(' '.join(words[:start])) + (1 if start > 0 else 0)
            char_end = char_start + len(chunk_text)

            chunks.append((chunk_text, char_start, char_end))

            # Move window with overlap
            start = end - overlap_words if end < len(words) else len(words)

        return chunks

    def chunk_constitution(
        self,
        text: str,
        country: str,
        source_url: str = ""
    ) -> Generator[Chunk, None, None]:
        """
        Chunk a constitutional document.

        Args:
            text: The raw constitutional text (may contain HTML)
            country: Country name for identification
            source_url: Source URL for metadata

        Yields:
            Chunk objects ready for embedding
        """
        # Clean the text
        clean_text = self._clean_html(text)

        chunk_index = 0

        if self.strategy == "hybrid":
            # Hybrid strategy: article-aware with size limits

            # First, extract and yield preamble if present
            preamble = self._extract_preamble(clean_text)
            if preamble:
                preamble_chunks = self._chunk_by_sentences(
                    preamble['content'],
                    self.max_chunk_size
                )
                for i, chunk_text in enumerate(preamble_chunks):
                    if self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                        yield Chunk(
                            chunk_id=self._generate_chunk_id(country, chunk_text, chunk_index),
                            content=chunk_text,
                            content_type="preamble",
                            source_document=source_url,
                            section_title="Preamble",
                            section_number=f"P.{i+1}" if len(preamble_chunks) > 1 else "P",
                            start_position=preamble['start'],
                            end_position=preamble['end'],
                            token_estimate=self._estimate_tokens(chunk_text),
                            metadata={"country": country, "chunk_part": i+1, "total_parts": len(preamble_chunks)}
                        )
                        chunk_index += 1

            # Extract articles
            articles = self._extract_articles(clean_text)

            if articles:
                # Process each article
                for i, article in enumerate(articles):
                    # Get article content (text until next article or end)
                    start = article['end']
                    end = articles[i + 1]['start'] if i + 1 < len(articles) else len(clean_text)
                    article_content = clean_text[start:end].strip()

                    if not article_content:
                        continue

                    # Chunk the article content
                    article_chunks = self._chunk_by_sentences(
                        article_content,
                        self.max_chunk_size
                    )

                    for j, chunk_text in enumerate(article_chunks):
                        if self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                            section_title = f"{article['type']} {article['number']}"
                            if article['title']:
                                section_title += f": {article['title']}"

                            yield Chunk(
                                chunk_id=self._generate_chunk_id(country, chunk_text, chunk_index),
                                content=chunk_text,
                                content_type=article['type'].lower(),
                                source_document=source_url,
                                section_title=section_title,
                                section_number=article['number'],
                                start_position=start,
                                end_position=end,
                                token_estimate=self._estimate_tokens(chunk_text),
                                metadata={
                                    "country": country,
                                    "article_type": article['type'],
                                    "article_number": article['number'],
                                    "chunk_part": j+1,
                                    "total_parts": len(article_chunks)
                                }
                            )
                            chunk_index += 1
            else:
                # No articles found, use sliding window
                for chunk_text, start, end in self._sliding_window_chunks(clean_text):
                    if self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                        yield Chunk(
                            chunk_id=self._generate_chunk_id(country, chunk_text, chunk_index),
                            content=chunk_text,
                            content_type="paragraph",
                            source_document=source_url,
                            start_position=start,
                            end_position=end,
                            token_estimate=self._estimate_tokens(chunk_text),
                            metadata={"country": country}
                        )
                        chunk_index += 1

        elif self.strategy == "sliding":
            # Pure sliding window approach
            for chunk_text, start, end in self._sliding_window_chunks(clean_text):
                if self._estimate_tokens(chunk_text) >= self.min_chunk_size:
                    yield Chunk(
                        chunk_id=self._generate_chunk_id(country, chunk_text, chunk_index),
                        content=chunk_text,
                        content_type="paragraph",
                        source_document=source_url,
                        start_position=start,
                        end_position=end,
                        token_estimate=self._estimate_tokens(chunk_text),
                        metadata={"country": country}
                    )
                    chunk_index += 1

        elif self.strategy == "paragraph":
            # Paragraph-based chunking
            paragraphs = clean_text.split('\n\n')
            position = 0

            for para in paragraphs:
                para = para.strip()
                if para and self._estimate_tokens(para) >= self.min_chunk_size:
                    # Further split if too large
                    if self._estimate_tokens(para) > self.max_chunk_size:
                        sub_chunks = self._chunk_by_sentences(para, self.max_chunk_size)
                        for sub_chunk in sub_chunks:
                            if self._estimate_tokens(sub_chunk) >= self.min_chunk_size:
                                yield Chunk(
                                    chunk_id=self._generate_chunk_id(country, sub_chunk, chunk_index),
                                    content=sub_chunk,
                                    content_type="paragraph",
                                    source_document=source_url,
                                    start_position=position,
                                    end_position=position + len(sub_chunk),
                                    token_estimate=self._estimate_tokens(sub_chunk),
                                    metadata={"country": country}
                                )
                                chunk_index += 1
                    else:
                        yield Chunk(
                            chunk_id=self._generate_chunk_id(country, para, chunk_index),
                            content=para,
                            content_type="paragraph",
                            source_document=source_url,
                            start_position=position,
                            end_position=position + len(para),
                            token_estimate=self._estimate_tokens(para),
                            metadata={"country": country}
                        )
                        chunk_index += 1

                position += len(para) + 2  # +2 for \n\n


def chunk_all_constitutions(db_path: str) -> Generator[tuple, None, None]:
    """
    Generator that yields (country, chunk) for all constitutions in the database.

    Args:
        db_path: Path to the constitutions database

    Yields:
        Tuples of (country_name, Chunk)
    """
    import sqlite3

    chunker = ConstitutionChunker(strategy="hybrid")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT country, full_text, source_url
        FROM constitutions
        ORDER BY country
    """)

    for row in cursor.fetchall():
        country = row['country']
        full_text = row['full_text']
        source_url = row['source_url'] or ''

        print(f"Chunking: {country}...")

        for chunk in chunker.chunk_constitution(full_text, country, source_url):
            yield (country, chunk)

    conn.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '..')

    # Test with a sample text
    sample_constitution = """
    PREAMBLE

    We the People of the United States, in Order to form a more perfect Union,
    establish Justice, insure domestic Tranquility, provide for the common defense,
    promote the general Welfare, and secure the Blessings of Liberty to ourselves
    and our Posterity, do ordain and establish this Constitution for the United
    States of America.

    Article I

    Section 1
    All legislative Powers herein granted shall be vested in a Congress of the
    United States, which shall consist of a Senate and House of Representatives.

    Section 2
    The House of Representatives shall be composed of Members chosen every second
    Year by the People of the several States.

    Article II

    Section 1
    The executive Power shall be vested in a President of the United States of
    America. He shall hold his Office during the Term of four Years.
    """

    chunker = ConstitutionChunker(max_chunk_size=200, min_chunk_size=50)

    print("Testing chunker with sample constitution:")
    print("=" * 60)

    for chunk in chunker.chunk_constitution(sample_constitution, "Test Country"):
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Type: {chunk.content_type}")
        print(f"Section: {chunk.section_title or 'N/A'}")
        print(f"Tokens: ~{chunk.token_estimate}")
        print(f"Content: {chunk.content[:100]}...")
