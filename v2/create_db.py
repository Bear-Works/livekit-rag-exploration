from sqlalchemy.ext.asyncio import create_async_engine
from model import *

# How to run async function:
# import asyncio

# async def my_async_function():
#     print("Starting async function")
#     await asyncio.sleep(1)  # Simulate an I/O-bound operation
#     print("Async function finished")

# if __name__ == "__main__":
#     print("Running the async function...")
#     asyncio.run(my_async_function())
#     print("Async function execution complete.")


# https://codeawake.com/blog/postgresql-vector-database
DB_URL = 'postgresql+asyncpg://awang@localhost:5432/rag_db'

engine = create_async_engine(DB_URL)

async def db_create():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

from sqlalchemy.ext.asyncio import async_sessionmaker
Session = async_sessionmaker(engine, expire_on_commit=False)

async def extract_text(docpath):
    with open(docpath, 'r') as file:
        content = file.read()
    return content

class TextSplitter:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
        self.splitters = [
            partial(split_by_separator, sep='\n\n'),
            partial(split_by_separator, sep='\n'),
            split_sentences,
            partial(split_by_separator, sep=' ')
        ]
    
    def split(self, text):
        splits = self._split_recursive(text)
        chunks = self._merge_splits(splits)
        return chunks 
        
async def add_document_to_vector_db(doc_path):
    text = extract_text(doc_path)
    doc_name = os.path.splitext(os.path.basename(doc_path))[0]
    
    chunks = []
    text_splitter = TextSplitter(chunk_size=512)
    text_chunks = text_splitter.split(text)
    for idx, text_chunk in enumerate(text_chunks):
        chunks.append({
            'text': text_chunk,
            'metadata_': {'doc': doc_name, 'index': idx}
        })

    vectors = await create_embeddings([chunk['text'] for chunk in chunks])

    for chunk, vector in zip(chunks, vectors):
        chunk['vector'] = vector
    
    async with Session() as db:
        for chunk in chunks:
            db.add(Vector(**chunk))
        await db.commit()

from openai import AsyncOpenAI
import os
client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

async def get_embeddings(input):
    res = await client.embeddings.create(input=input, model='text-embedding-3-large', dimensions=1024)
    return [item.embedding for item in res.data]