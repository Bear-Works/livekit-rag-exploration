# https://github.com/codemaker2015/vector-database-experiments/blob/master/pgvector/app.py
from openai import OpenAI
from pgvector.psycopg import register_vector
import psycopg
import os

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# DB_URL = 'postgresql+asyncpg://awang@localhost:5432/rag_db'
conn = psycopg.connect(
    host='localhost',
    port=5432,
    user='awang',
    # password='password',
    dbname='rag_db',
    autocommit=True)

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS documents')
conn.execute('CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(1536))')

input = [
    'The dog is barking',
    'The cat is purring',
    'The bear is growling',
    'The bird is singing',
    'The elephant is trumpeting',
    'The lion is roaring',
    'The horse is neighing',
    'The monkey is chattering'
]


# os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'
# if os.getenv("OPENAI_API_KEY") is not None:
#     openai.api_key = os.getenv("OPENAI_API_KEY")
#     print ("OPENAI_API_KEY is ready")
# else:
#     print ("OPENAI_API_KEY environment variable not found")

from openai.types import CreateEmbeddingResponse, Embedding, EmbeddingModel

# client.embeddings.create(**params) -> CreateEmbeddingResponse
# response = openai.Embedding.create(input=input, model='text-embedding-ada-002')
# embeddings = [v['embedding'] for v in response['data']]
response = client.embeddings.create(input=input, model='text-embedding-ada-002')
# response.data[0].embedding
embeddings = [v.embedding for v in response.data]


for content, embedding in zip(input, embeddings):
    conn.execute('INSERT INTO documents (content, embedding) VALUES (%s, %s)', (content, embedding))

document_id = 1
neighbors = conn.execute('SELECT content FROM documents WHERE id != %(id)s ORDER BY embedding <=> (SELECT embedding FROM documents WHERE id = %(id)s) LIMIT 5', {'id': document_id}).fetchall()
for neighbor in neighbors:
    print(neighbor[0])