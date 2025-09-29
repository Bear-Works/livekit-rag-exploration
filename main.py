#!/usr/bin/env python3
"""
RAG-enabled voice agent example for LiveKit Agents 1.0

This agent uses the RAG (Retrieval Augmented Generation) plugin to provide
information from a knowledge base when answering user questions.

Before running this agent:
1. Make sure you have your OpenAI API key in a .env file
2. Run build_rag_data.py to build the RAG database
"""
import os
import time
import aiohttp
from aiohttp import web
import aioboto3
from botocore.exceptions import ClientError
import logging
import pickle
from pathlib import Path
from typing import Literal, Any, AsyncIterable
from collections.abc import Iterable
from dataclasses import dataclass
from dotenv import load_dotenv
import annoy
from livekit import rtc

from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    RunContext,
    function_tool,
    RoomInputOptions,
    Agent,
    AgentSession,
    ModelSettings,
)
from livekit.plugins import openai, silero, deepgram, noise_cancellation
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.agents.voice.transcription.filters import filter_markdown #in tts_node

# Redis server
import asyncio
import redis
from redis.asyncio import Redis

# For tts input -> redis key
import json, hashlib
import re
import numpy as np 
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Frames --> WAV file --> S3 (to send to S3)
import wave
import boto3
from uuid import uuid4

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag-agent")

# RAG Index Types and Classes
Metric = Literal["angular", "euclidean", "manhattan", "hamming", "dot"]
ANNOY_FILE = "index.annoy"
METADATA_FILE = "metadata.pkl"

# Set up redis server
r = redis.Redis.from_url("redis://127.0.0.1:6380/0", decode_responses=True)

# Redis cache key
_WS_RE = re.compile(r"\s+") # whitespace
_PUNCT_RE = re.compile(r"[.,!?;:—\-–…'\"`“”’(){}\[\]]+") # punctutation

# Pre-trained embedding model 
emb_model = SentenceTransformer("all-MiniLM-L6-v2") 

# Removing stop words (filler words) - can add more filler words if needed
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
stop_words = set(stopwords.words("english"))

# RediSearch constant variables
INDEX_NAME = "idx::tts"         # RediSearch index for TTS cache
VEC_FIELD = "embedding"               # Field in hash that stores embedding
EMB_DIM = 384                   # Dimensions of embedding model

# Set s3 client for storing audio files
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-1"
)

BUCKET_NAME = "livekit-audio-cache"

@dataclass
class _FileData:
    f: int
    metric: Metric
    userdata: dict[int, Any]


@dataclass
class Item:
    i: int
    userdata: Any
    vector: list[float]


@dataclass
class QueryResult:
    userdata: Any
    distance: float


class AnnoyIndex:
    def __init__(self, index: annoy.AnnoyIndex, filedata: _FileData) -> None:
        self._index = index
        self._filedata = filedata

    @classmethod
    def load(cls, path: str) -> "AnnoyIndex":
        p = Path(path)
        index_path = p / ANNOY_FILE
        metadata_path = p / METADATA_FILE

        with open(metadata_path, "rb") as f:
            metadata: _FileData = pickle.load(f)

        index = annoy.AnnoyIndex(metadata.f, metadata.metric)
        index.load(str(index_path))
        return cls(index, metadata)

    @property
    def size(self) -> int:
        return self._index.get_n_items()

    def items(self) -> Iterable[Item]:
        for i in range(self._index.get_n_items()):
            item = Item(
                i=i,
                userdata=self._filedata.userdata[i],
                vector=self._index.get_item_vector(i),
            )
            yield item

    def query(
        self, vector: list[float], n: int, search_k: int = -1
    ) -> list[QueryResult]:
        ids = self._index.get_nns_by_vector(
            vector, n, search_k=search_k, include_distances=True
        )
        return [
            QueryResult(userdata=self._filedata.userdata[i], distance=distance)
            for i, distance in zip(*ids)
        ]


# Making a looser key for more hits
def normalize(s: str) -> str:
    s = s.strip()               # Removes leading and trailing whitespaces
    s = _WS_RE.sub (" ", s)     # Remove whitespace
    s = _PUNCT_RE.sub(" ", s)   # Remove punctuation
    s = s.lower()               # Lowercase 
    logger.info(f"normalize function: {s}")
    return s

# Remove filler words 
def remove_filler(s: str) -> str:
    tokens = word_tokenize(s)
    filtered_tokens = [t for t in tokens if t not in stop_words]
    # logger.info(f"remove_filler function: {" ".join(filtered_tokens)}")
    return " ".join(filtered_tokens)

# Embed the string --> vector that represents the meaning of the string
def embed_text(text: str) -> np.ndarray:
    vec = emb_model.encode([text], normalize_embeddings=True)[0]  # L2-normalized
    return vec.astype(np.float32)  # Redis expects float32

# Compare the query string with the hit and compare how similar they are using cosine similarity
# Return url, dist, if they are similar
def similar(res: list):
    THRESHOLD_DIST = 0.3

    if not res or res[0] == 0:
        return None, None, False

    # res format: [count, key, [field, value, field, value, ...]]
    fields = dict(zip(res[2][0::2], res[2][1::2]))
    s3_key = fields.get("s3_key")
    dist = float(fields.get("dist"))
    # cos_sim = 1 - dist # DO I USE THIS ?!

    similar = dist <= THRESHOLD_DIST
    return s3_key, dist, similar
 
# Search cache to see if there are any hits
# Return the res, vec (embedding)
def search_cache(text: str):
    # Remove whitespace/ newlines from text
    normalized_text = normalize(text)

    # Remove filler words 
    filtered_text = remove_filler(normalized_text)

    # Embed the normalized text
    vec = embed_text(filtered_text)

    # Run KNN (k-nearest neighbors) on vec with RediSearch
    assert vec.dtype == np.float32 and vec.shape == (384,)
    query = f"*=>[KNN 1 @{VEC_FIELD} $vec AS dist]"

    # Res is a list 
    res = r.execute_command(
        "FT.SEARCH", INDEX_NAME,
        query,                    
        "PARAMS", "2", "vec", vec.tobytes(),
        "SORTBY", "dist",
        "RETURN", "2", "s3_key", "dist",  
        "DIALECT", "2"
    )

    # RediSearch Index
    # r.execute_command(
    #     "FT.CREATE", INDEX_NAME,
    #     "ON", "HASH",
    #     "PREFIX", "1", "audio",
    #     "SCHEMA",
    #     "s3_key", "TEXT",
    #     "transcript", "TEXT",
    #     "embedding", "VECTOR", "HNSW", "6",
    #         "TYPE", "FLOAT32",
    #         "DIM", str(384),
    #         "DISTANCE_METRIC", "COSINE"
    # )

    logger.info(f"query: {query}")
    logger.info(f"search_cache function: {res}")

    return vec, res

# Create WAV file and upload to S3
def upload(frames: list, filename: str):
    # Frames --> WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)    
        wf.setsampwidth(2)        
        wf.setframerate(24000)      # OpenAI TTS models return 24000 Hz PCM16
        for frame in frames:
            wf.writeframes(frame.data) 

    # Upload to S3
    s3.upload_file(filename, BUCKET_NAME, filename)

    # Delete extra wave file from project directory
    if os.path.exists(filename):
        os.remove(filename)

# Create a new cache entry in redis 
def add_entry(s3_key: str, transcript: str, embedding: np.ndarray):
    logger.info(f"In the add entry function. This is s3 uri: {s3_key}, transcript: {transcript}")
    # Make sure embedding is in the correct format (dtype/dshape)
    # emb = np.asarray(embedding, dtype=np.float32)
    logger.info(f"embed shape:  {embedding.shape}")
    # assert emb.shape == (384,), f"Expected (384,), got {emb.shape}"

    key = f"audio:{uuid4()}"        
    r.hset(key, mapping={
        b"s3_key": s3_key.encode("utf-8"),
        b"transcript": transcript.encode("utf-8"),
        b"embedding": embedding.tobytes(), 
    })

# Make s3 presigned url for streaming
def make_audio_url(key: str, expires_in: int = 3600) -> str:
    s3 = boto3.client("s3")
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": key},
        ExpiresIn=expires_in,
    )

# Stream the audio from url
async def stream(url: str):
    logger.info("The streaming has started.")
    header_bytes = 44      
    sr = 24000
    ch = 1
    sampwidth = 2            
    target_ms = 20

    samples_per_chunk = int(sr * (target_ms / 1000.0))
    bytes_per_sample = ch * sampwidth
    bytes_per_chunk = samples_per_chunk * bytes_per_sample
    buffer = b""

    logger.info(f"GET url: {url}")
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            logger.info(f"HTTP {resp.status} {resp.headers.get('Content-Type')}")
            resp.raise_for_status()

            # 1) Skip (simple) WAV header
            await resp.content.readexactly(header_bytes)

            # 2) Read & yield fixed-size chunks
            while True:
                chunk = await resp.content.read(8192)
                if not chunk:
                    # EOF — stop reading, then flush remainder once below
                    break

                buffer += chunk
                while len(buffer) >= bytes_per_chunk:
                    payload = buffer[:bytes_per_chunk]
                    buffer = buffer[bytes_per_chunk:]
                    yield rtc.AudioFrame(
                        data=payload,
                        sample_rate=sr,
                        num_channels=ch,
                        samples_per_channel=samples_per_chunk,
                    )

            # 3) Flush any tail once after EOF (truncate to whole samples)
            if buffer:
                usable = len(buffer) - (len(buffer) % bytes_per_sample)
                if usable > 0:
                    tail = buffer[:usable]
                    tail_samples = usable // bytes_per_sample
                    yield rtc.AudioFrame(
                        data=tail,
                        sample_rate=sr,
                        num_channels=ch,
                        samples_per_channel=tail_samples,
                    )
                # drop any sub-sample residue silently

# Yield the string so can iterate through it again in tts_node
async def yield_s(s: str):
    yield s


class RAGEnrichedAgent(Agent):
    """
    An agent that can answer questions using RAG (Retrieval Augmented Generation).
    """

    def __init__(self) -> None:
        """Initialize the RAG-enabled agent."""
        super().__init__(
            instructions="""
                You are a helpful voice assistant specializing in knowledge about a Shopify store called Sirius Dice
                You can answer questions about Sirius Dice, their products, the prices of their products, etc.
                Your responses should always be concise and suitable for text-to-speech output, so be casual and avoid using markdown or other special formatting.
            """,
        )

        # Initialize RAG components
        vdb_dir = Path(__file__).parent / "data"
        data_path = vdb_dir / "paragraphs.pkl"

        if not vdb_dir.exists() or not data_path.exists():
            logger.warning(
                "RAG database not found. Please run build_rag_data.py first:\n"
                "$ python build_rag_data.py"
            )
            return

        # Load RAG index and data
        self._index_path = vdb_dir
        self._data_path = data_path
        self._embeddings_dimension = 1536
        self._embeddings_model = "text-embedding-3-small"
        self._seen_results = set()  # Track previously seen results

        try:
            self._annoy_index = AnnoyIndex.load(str(self._index_path))
            with open(self._data_path, "rb") as f:
                self._paragraphs_by_uuid = pickle.load(f)
            logger.info("RAG database loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load RAG database: {e}")
    
    

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        # Combine the text into a string 
        chunks = []
        transcript = ""
        async for chunk in text:
            chunks.append(chunk)
            transcript = "".join(chunks)
            logger.info("TTS input: %s", transcript)

        # Create embedding for string and search in cache
        vec, res = search_cache(transcript)
        s3_key, dist, is_similar = similar(res)

        logger.info(f"Similarity: {dist} so {is_similar} and key is {s3_key}")

        # Embeddings are similar: get s3 uri --> stream from s3
        if is_similar:
            url = make_audio_url(s3_key)
            logger.info(f"The audio url: {url}")
            
            got = 0
            async for frame in stream(url):
                got += 11
                yield frame
            logger.info(f"cache stream yield {got} frames")
            return
        
        # Embeddings not similar: generate audio -> store in s3 -> new cache entry -> play audio
        else:
            src = yield_s(transcript)
            frames = []

            async for frame in Agent.default.tts_node(self, src, model_settings):
                logger.info(f"Frame bytes: {len(frame.data)}")
                frames.append(frame)
                yield frame

            # Generate unique filename for audio file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            unique_id = uuid4().hex[:6]
            filename = f"tts_{timestamp}_{unique_id}.wav"

            # Convert frames to WAV, then upload to S3
            upload(frames, filename)

            # Add a new redis cache entry
            add_entry(filename, transcript, vec)


    @function_tool
    async def livekit_docs_search(self, context: RunContext, query: str):
        """
        Lookup information in the Sirius Dice database. Will not return results already returned in previous lookups.
        Questions will usually be about products, pricing, and suggestions. 
        """
        logger.info("[RAG] tool called with query: %s", query)

        try:
            # Generate embeddings for the query
            query_embedding = await openai.create_embeddings(
                input=[query],
                model=self._embeddings_model,
                dimensions=self._embeddings_dimension,
            )

            # Query the index for more results than we need to ensure we have enough new content
            all_results = self._annoy_index.query(
                query_embedding[0].embedding, n=5
            )  # Get more results initially

            # Filter out previously seen results
            new_results = [
                r for r in all_results if r.userdata not in self._seen_results
            ]

            # If we don't have enough new results, clear the seen results and start fresh
            if len(new_results) == 0:
                return "No new results found."
            else:
                new_results = new_results[:2]  # Take top 2 new results

            # Build context from multiple relevant paragraphs
            context_parts = []
            for result in new_results:
                # Add result to seen set
                self._seen_results.add(result.userdata)

                paragraph = self._paragraphs_by_uuid.get(result.userdata, "")
                if paragraph:
                    # Extract source URL if available in the paragraph
                    source = "Unknown source"
                    if "from [" in paragraph:
                        source = paragraph.split("from [")[1].split("]")[0]
                        paragraph = paragraph.split("]")[1].strip()

                    context_parts.append(f"Source: {source}\nContent: {paragraph}\n")

            if not context_parts:
                return

            # Combine all context parts with clear separation
            full_context = "\n\n".join(context_parts)
            # logger.info(
            #     f"Results for query: {query}, full context: {full_context.replace('\n', '\\n')}"
            # )

            return full_context
        except Exception as e:
            return "Could not find any relevant information for that query."

    async def on_enter(self):
        """Called when the agent enters the session."""
        self.session.generate_reply(
            instructions="Briefly greet the user and offer your assistance with Sirius Dice."
        )


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the agent."""

    session = AgentSession(
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(
            instructions="You are a helpful assistant with a pleasant voice.",
            voice="ash",
        ),
        turn_detection=EnglishModel(),
        vad=silero.VAD.load(),
    )

    await session.start(
        agent=RAGEnrichedAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )



   
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


