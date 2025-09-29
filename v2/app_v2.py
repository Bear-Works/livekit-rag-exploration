# Process scraped Sirius Dice website data from build_rag_data.py output and create embeddings for postgres storage
from openai import OpenAI
from pgvector.psycopg import register_vector
import psycopg
import os
import json
from pathlib import Path
import re
from html import unescape
import argparse

# Documents: contains JSON product data (/Users/awang/dev/bearworks/rag/data/sirius_raw_data.json)
# Documents2: contains text scraped website (/Users/awang/dev/bearworks/livekit/python-agents-examples/rag/data/raw_data-awesomedice.txt)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process website data and create embeddings for RAG')
parser.add_argument('--create-index', action='store_true', default=False,
                    help='Create database and index documents (default: False - only run similarity search)')
args = parser.parse_args()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Database connection
conn = psycopg.connect(
    host='localhost',
    port=5432,
    user='awang',
    dbname='rag_db',
    autocommit=True
)

# Setup database
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

if args.create_index:
    # Create table for website documents (policies, about, blog posts)
    print("Creating database table...")
    conn.execute('DROP TABLE IF EXISTS documents2')
    conn.execute('''
        CREATE TABLE documents2 (
            id bigserial PRIMARY KEY,
            url text,
            title text,
            content text,
            embedding vector(1536)
        )
    ''')
else:
    print("Skipping database creation (use --create-index to create and populate database)")

def clean_html(html_text):
    """Remove HTML tags and clean up text for embedding"""
    if not html_text:
        return ""

    # Remove HTML tags
    clean_text = re.sub('<[^<]+?>', '', html_text)
    # Unescape HTML entities
    clean_text = unescape(clean_text)
    # Clean up whitespace
    clean_text = ' '.join(clean_text.split())
    return clean_text

def prepare_content_for_embedding(item):
    """Extract and clean content from scraped website data"""
    # Handle different possible structures in the scraped data
    title = item.get('title', '')
    content = ""

    # Check for different content fields that might exist
    if 'html' in item:
        content = clean_html(item['html'])
    elif 'content' in item:
        # For text format, content might already be clean
        if '<' in item['content'] and '>' in item['content']:
            content = clean_html(item['content'])
        else:
            content = item['content']
    elif 'text' in item:
        content = item['text']
    elif 'body' in item:
        content = clean_html(item['body'])

    # Combine title and content
    if title and content:
        full_content = f"{title}. {content}"
    elif title:
        full_content = title
    elif content:
        full_content = content
    else:
        return None

    return full_content.strip()

if args.create_index:
    # Load data - support both JSON and text formats
    json_path = Path(__file__).parent.parent / 'data' / 'sirius_raw_data.json'
    text_path = Path('/Users/awang/dev/bearworks/livekit/python-agents-examples/rag/data/raw_data-awesomedice.txt')

    website_data = []

    # Try JSON format first
    if False and json_path.exists():
        print(f"Loading JSON data from {json_path}")
        with open(json_path, 'r') as f:
            website_data = json.load(f)
        print(f"Found {len(website_data)} items in JSON format")

    # Also try text format
    elif text_path.exists():
        print(f"Loading text data from {text_path}")
        with open(text_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        # Split text into paragraphs/sections
        # Assuming paragraphs are separated by double newlines
        paragraphs = [p.strip() for p in raw_text.split('\n\n') if p.strip()]

        # Convert to JSON-like format for consistent processing
        website_data = []
        for i, paragraph in enumerate(paragraphs):
            # Try to extract title from first line if it looks like a header
            lines = paragraph.split('\n')
            if len(lines) > 1 and len(lines[0]) < 100:  # Likely a title
                title = lines[0].strip()
                content = '\n'.join(lines[1:]).strip()
            else:
                title = f"Section {i+1}"
                content = paragraph

            website_data.append({
                'title': title,
                'content': content,
                'url': f"text-section-{i+1}"
            })

        print(f"Found {len(website_data)} paragraphs in text format")

    else:
        print(f"Error: Neither {json_path} nor {text_path} found.")
        print("Please ensure you have data in one of these formats.")
        exit(1)

    if not website_data:
        print("No data found to process.")
        exit(1)

if args.create_index:
    # Process items in batches for efficiency
    batch_size = 20
    total_processed = 0

    # Load items to DB for embedding
    for i in range(0, len(website_data), batch_size):
        batch = website_data[i:i + batch_size]

        # Prepare content for embedding
        batch_content = []
        batch_items = []

        for item in batch:
            content = prepare_content_for_embedding(item)
            if content and len(content.strip()) > 10:  # Only process meaningful content
                batch_content.append(content)
                batch_items.append(item)

        if not batch_content:
            continue

        print(f"Creating embeddings for batch {i//batch_size + 1} ({len(batch_content)} items)")

        # Create embeddings for the batch
        response = client.embeddings.create(
            input=batch_content,
            model='text-embedding-3-small'
        )

        # Insert into database
        for item, content, embedding_data in zip(batch_items, batch_content, response.data):
            conn.execute('''
                INSERT INTO documents2 (url, title, content, embedding)
                VALUES (%s, %s, %s, %s)
            ''', (
                item.get('url', ''),
                item.get('title', ''),
                content,
                embedding_data.embedding
            ))

            total_processed += 1

        print(f"Processed {total_processed}/{len(website_data)} items")

    print(f"\nSuccessfully processed {total_processed} website documents!")
else:
    print("Skipping data indexing (use --create-index to populate database)")

# Test similarity search
print("\nTesting similarity search...")
# test_query = "shipping policy"
# test_query = "how can i return an item?"
test_query = "i'm looking for a gift.  recommend me a set"
query_response = client.embeddings.create(
    input=[test_query],
    model='text-embedding-3-small'
)

results = conn.execute('''
    SELECT title, url, content
    FROM documents2
    ORDER BY embedding <=> %s::vector
    LIMIT 3
''', (query_response.data[0].embedding,)).fetchall()

print(results)
print(f"\nTop 3 results for '{test_query}':")
for i, (title, url, content) in enumerate(results, 1):
    print(f"{i}. {title}")
    print(f"   URL: {url}")
    print(f"   Content: {content[:150]}...")
    print()

conn.close()
print("Database connection closed.")
