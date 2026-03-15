import os
import re
import json
import threading
from contextlib import asynccontextmanager
from collections import deque
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
)

load_dotenv()

DATA_DIR = "./data"
VECTOR_DIR = os.path.join(DATA_DIR, "chroma")
SCHEMA_DIR = os.path.join(DATA_DIR, "transcripts")
os.makedirs(VECTOR_DIR, exist_ok=True)
os.makedirs(SCHEMA_DIR, exist_ok=True)

EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

LLM = ChatGroq(
    model="llama-3.3-70b-versatile",  # valid Groq model
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=1,
    max_tokens=2048
)

MAX_MEMORY_TURNS = 6

# ---------------- STATE ----------------
VECTOR_STORES = {}
INDEX_STATUS = {}
LOCKS = {}
MEMORY_POOL = {}


# ---------------- LIFESPAN ----------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_existing_indexes()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"]
)


# ---------------- SCHEMAS ----------------
class IngestURL(BaseModel):
    url: str

class Ask(BaseModel):
    video_id: str
    question: str
    session_id: str | None = None

class CaptionRequest(BaseModel):
    prompt: str

class CaptionResponse(BaseModel):
    youtube_caption: str

class ThumbnailRequest(BaseModel):
    prompt: str

class ThumbnailResponse(BaseModel):
    suggestions: list[str]

class SummaryRequest(BaseModel):
    video_id: str
    max_segments: int = 8

class SummaryResponse(BaseModel):
    summary: str

class SimilarityRequest(BaseModel):
    video_id: str
    query: str
    top_k: int = 10


# ---------------- HELPERS ----------------
def extract_video_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    return None


def fetch_transcript(video_id: str):
    try:
        return YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        raise ValueError("Transcript unavailable")  # plain exception, safe in threads


def get_store(video_id: str):
    return Chroma(
        collection_name=f"video_{video_id}",
        embedding_function=EMBEDDINGS,
        persist_directory=os.path.join(VECTOR_DIR, video_id)
    )


def get_memory(session_id: str, video_id: str):
    key = f"{session_id}::{video_id}"  # "::" avoids underscore collision
    if key not in MEMORY_POOL:
        MEMORY_POOL[key] = deque(maxlen=MAX_MEMORY_TURNS)
    return MEMORY_POOL[key]


def extract_json(text: str) -> dict:
    """Safely extract JSON from LLM response even if it contains markdown fences."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")
    return json.loads(match.group())


# ---------------- INDEXING ----------------
def index_video(video_id: str):
    LOCKS.setdefault(video_id, threading.Lock())

    with LOCKS[video_id]:
        if INDEX_STATUS.get(video_id, {}).get("status") == "indexed":
            return

        INDEX_STATUS[video_id] = {"status": "indexing", "chunk_count": 0}

        try:
            transcript = fetch_transcript(video_id)

            group_size = 10
            grouped_texts, grouped_metadatas = [], []

            for i in range(0, len(transcript), group_size):
                group = transcript[i:i + group_size]
                text = " ".join(c.text for c in group)
                metadata = {
                    "video_id": video_id,
                    "start": group[0].start,
                    "duration": sum(c.duration for c in group)
                }
                grouped_texts.append(text)
                grouped_metadatas.append(metadata)

            splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=160)
            docs = []
            for text, meta in zip(grouped_texts, grouped_metadatas):
                docs.extend(splitter.create_documents([text], metadatas=[meta]))

            texts, metadatas, ids = [], [], []
            for i, d in enumerate(docs):
                chunk_id = f"{video_id}_vec_{i}"
                meta = dict(d.metadata)
                meta["embedding_id"] = chunk_id
                meta["chunk_index"] = i
                texts.append(d.page_content)
                metadatas.append(meta)
                ids.append(chunk_id)

            store = get_store(video_id)
            store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            VECTOR_STORES[video_id] = store
            INDEX_STATUS[video_id] = {"status": "indexed", "chunk_count": len(texts)}

        except Exception as e:
            print(f"[index_video] Failed for {video_id}: {e}")
            INDEX_STATUS[video_id] = {"status": "failed", "chunk_count": 0}


# ---------------- STARTUP ----------------
def load_existing_indexes():
    for vid in os.listdir(VECTOR_DIR):
        try:
            VECTOR_STORES[vid] = get_store(vid)
            INDEX_STATUS[vid] = {"status": "indexed", "chunk_count": 0}
        except Exception as e:
            print(f"[startup] Could not load index for {vid}: {e}")


# ---------------- PROMPT ----------------
PROMPT = PromptTemplate(
    template="""
SYSTEM ROLE:
You are a **YouTube video–grounded assistant**.
Your primary job is to answer questions using the provided transcript context.
You may also respond politely to basic conversational messages (e.g., greetings).

CORE KNOWLEDGE RULES:
- Use the provided transcript context as the main source of truth.
- Do NOT invent facts, names, or explanations.
- Do NOT fabricate timestamps or details not present.
- If the transcript does not contain the answer, say:
"I don't know based on this video."

REASONABLE FLEXIBILITY:
- If the user says "hi", "hello", or similar greetings, respond briefly and politely.
- If the user asks about your role or capability, explain it simply (no technical jargon).
- Do NOT answer factual or technical questions without transcript support.

ANSWERING BEHAVIOR:
- Be clear, helpful, and readable.
- Explain only what is explicitly stated in the transcript.
- If a "why" or "how" is not explained in the transcript, say you don't know.
- If multiple viewpoints exist in the transcript, present them neutrally.

CHAT HISTORY USAGE:
- Chat history is for conversational continuity only.
- Do NOT use chat history as a factual source.
- If chat history conflicts with transcript, transcript always takes priority.

TIMESTAMPS:
- Use timestamps only if they are present in the transcript context.
- Never invent timestamps.

TONE & STYLE:
- Natural, calm, and professional.
- No emojis. No hype. No unnecessary apologies. No speculation.
- Short paragraphs or bullet points when useful.

FAILURE MODE:
If the transcript context does NOT contain the information required to answer a factual question, respond exactly with:
"I don't know based on this video."

--------------------
CHAT HISTORY:
{chat_history}

--------------------
TRANSCRIPT CONTEXT:
{context}

--------------------
USER QUESTION:
{question}

--------------------
FINAL ANSWER:
""",
    input_variables=["context", "question", "chat_history"]
)


# ---------------- ROUTES ----------------
@app.post("/ingest-url")
def ingest(payload: IngestURL):
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")

    if video_id not in VECTOR_STORES:
        threading.Thread(target=index_video, args=(video_id,), daemon=True).start()

    return {"video_id": video_id, "status": "indexing"}


@app.post("/ask")
def ask(payload: Ask):
    if payload.video_id not in VECTOR_STORES:
        status = INDEX_STATUS.get(payload.video_id, {}).get("status")
        if status != "indexing":
            threading.Thread(target=index_video, args=(payload.video_id,), daemon=True).start()
        return {"answer": "Indexing video, please wait a moment and ask again.", "references": []}

    store = VECTOR_STORES[payload.video_id]
    memory = get_memory(payload.session_id or "default", payload.video_id)

    retriever = store.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke(payload.question)

    context = "\n\n".join(d.page_content for d in docs)
    refs = [{"start": int(d.metadata["start"]), "duration": d.metadata["duration"]} for d in docs]

    chain = (
        RunnableParallel({
            "context": lambda _: context,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: "\n".join(memory)
        })
        | PROMPT
        | LLM
        | StrOutputParser()
    )

    answer = chain.invoke(payload.question)
    memory.append(f"Q: {payload.question}\nA: {answer}")

    return {"answer": answer, "references": refs}


@app.get("/status/{video_id}")
def status(video_id: str):
    return INDEX_STATUS.get(video_id, {"status": "not_indexed", "chunk_count": 0})


@app.get("/health")
def health():
    return {"videos": len(VECTOR_STORES), "sessions": len(MEMORY_POOL)}


# ---------------- CAPTION GENERATION ----------------
caption_prompt = PromptTemplate(
    template="""
You are a social media content expert.

User idea:
{user_prompt}

Generate a YouTube caption.

Rules:
- SEO optimized
- Slightly longer
- No hashtags inside sentence

Return strictly in JSON format:
{{
    "youtube_caption": "..."
}}
""",
    input_variables=["user_prompt"]
)

caption_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.6
)

caption_chain = caption_prompt | caption_model | StrOutputParser()


@app.post("/generate-caption", response_model=CaptionResponse)
async def generate_caption(request: CaptionRequest):
    try:
        result = caption_chain.invoke({"user_prompt": request.prompt})
        parsed = extract_json(result)
        return {"youtube_caption": parsed["youtube_caption"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- THUMBNAIL TEXT ----------------
thumbnail_prompt = PromptTemplate(
    template="""
You are a YouTube growth strategist.

User topic:
{user_prompt}

Generate short thumbnail text phrases.

Rules:
- 2 to 6 words per phrase
- Attention grabbing
- High curiosity
- Suitable for YouTube thumbnails
- No emojis, no hashtags

Return strictly JSON:
{{
  "suggestions": ["...", "...", "...", "...", "..."]
}}
""",
    input_variables=["user_prompt"]
)

thumbnail_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

thumbnail_chain = thumbnail_prompt | thumbnail_model | StrOutputParser()


@app.post("/thumbnail-text", response_model=ThumbnailResponse)
async def generate_thumbnail_text(request: ThumbnailRequest):
    try:
        result = thumbnail_chain.invoke({"user_prompt": request.prompt})
        parsed = extract_json(result)
        return {"suggestions": parsed["suggestions"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- SUMMARIZATION ----------------
summary_prompt = PromptTemplate(
    template="""
You are a precise summarization system.

Task:
Summarize the provided transcript segments.

Rules:
- Extract only key ideas.
- Do not invent information.
- Write concise and clear paragraphs.
- Focus on the main concepts discussed.

Transcript Segments:
{context}

Return only the summary text.
""",
    input_variables=["context"]
)

summary_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3
)

summary_chain = summary_prompt | summary_model | StrOutputParser()


@app.post("/summarize", response_model=SummaryResponse)
def summarize_video(payload: SummaryRequest):
    if payload.video_id not in VECTOR_STORES:
        raise HTTPException(404, "Video index not found")

    store = VECTOR_STORES[payload.video_id]

    # Retrieve evenly spaced chunks for better coverage
    all_docs = store.get()
    total = len(all_docs["documents"])
    if total == 0:
        raise HTTPException(404, "No content indexed for this video")

    step = max(1, total // payload.max_segments)
    selected_texts = all_docs["documents"][::step][:payload.max_segments]
    context = "\n\n".join(selected_texts)

    summary = summary_chain.invoke({"context": context})
    return {"summary": summary}


# ---------------- TRANSCRIPT SCHEMA ----------------
@app.post("/transcript-schema")
def transcript_schema(payload: IngestURL):
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(400, "Invalid YouTube URL")

    transcript = fetch_transcript(video_id)

    group_size = 10
    results = []

    for i in range(0, len(transcript), group_size):
        group = transcript[i:i + group_size]
        results.append({
            "video_id": video_id,
            "start_time": group[0].start,
            "end_time": group[-1].start + group[-1].duration,
            "text": " ".join(c.text for c in group),
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "updatedAt": datetime.now(timezone.utc).isoformat()
        })

    file_path = os.path.join(SCHEMA_DIR, f"{video_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also trigger ChromaDB indexing if not already done
    if video_id not in VECTOR_STORES:
        threading.Thread(target=index_video, args=(video_id,), daemon=True).start()

    return {"video_id": video_id, "saved_to": file_path, "segments": results}


# ---------------- SEARCH VIDEO ----------------

class GlobalSearchRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/search-video")
def search_video(payload: GlobalSearchRequest):
    if not VECTOR_STORES:
        raise HTTPException(404, "No videos indexed yet.")

    all_results = []

    for video_id, store in VECTOR_STORES.items():
        try:
            docs_scores = store.similarity_search_with_score(payload.query, k=payload.top_k)
            for doc, distance in docs_scores:
                all_results.append({
                    "video_id": video_id,
                    "embedding_id": doc.metadata.get("embedding_id"),
                    "similarity_score": round(1 - float(distance), 4),
                    "start_time": doc.metadata.get("start"),
                    "duration": doc.metadata.get("duration"),
                    "text": doc.page_content
                })
        except Exception as e:
            print(f"[search-video] Skipped {video_id}: {e}")
            continue

    if not all_results:
        return {"query": payload.query, "results": []}

    # sort by best similarity across all videos
    all_results.sort(key=lambda x: x["similarity_score"], reverse=True)

    return {
        "query": payload.query,
        "results": all_results[:payload.top_k]
    }


class Request(BaseModel):
    url: str

def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1)

def get_metadata(url):
    ydl = yt_dlp.YoutubeDL({"quiet": True})
    info = ydl.extract_info(url, download=False)

    return {
        "title": info["title"],
        "description": info["description"],
        "duration": info["duration"],
        "thumbnail": info["thumbnail"]
    }

def create_clips(transcript):

    clips = []
    window = 20

    for i in range(0, len(transcript), window):

        start = transcript[i]["start"]
        end = transcript[min(i+window-1, len(transcript)-1)]["start"]

        text = " ".join(t["text"] for t in transcript[i:i+window])

        score = min(100, 70 + len(text) % 30)

        clips.append({
            "id": f"clip-{i}",
            "title": text[:60],
            "start_time": int(start),
            "end_time": int(end),
            "score": score
        })

    return sorted(clips, key=lambda x: x["score"], reverse=True)[:10]


@app.post("/extract")
def extract(req: Request):

    video_id = extract_video_id(req.url)

    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    metadata = get_metadata(req.url)

    clips = create_clips(transcript)

    return {
        "title": metadata["title"],
        "description": metadata["description"],
        "duration": metadata["duration"],
        "platform": "YouTube",
        "embedUrl": f"https://www.youtube.com/embed/{video_id}",
        "thumbnailUrl": metadata["thumbnail"],
        "clips": clips
    }


from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

load_dotenv()

try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-preview-image-generation"
)

generation_config = {
    "response_modalities": ["TEXT", "IMAGE"]
}


class GenerateRequest(BaseModel):
    event_name: str = "TechXperts 2025"
    event_description: str = "A national-level hackathon bringing together innovators."


@app.post("/generate")
async def generate_image(payload: GenerateRequest):

    try:
        text_input = (
            f"A futuristic, collectible NFT memento token for the event: '{payload.event_name}'.\n\n"
            f"Core Concept: '{payload.event_description}'.\n\n"
            "Object & Form: A distinct 2D illustrated token, symbolic coin, crystal, holographic card, or futuristic emblem.\n"
            "Style: Futuristic, Web3, cyberpunk, neon gradients, vector-style, flat geometric.\n"
            "Background: Minimalist dark abstract background.\n"
            f"Text: Include '{payload.event_name}' subtly.\n"
            "Avoid: 3D renders, photorealism, blurry, cartoonish, watermark."
        )

        response = model.generate_content(
            contents=[text_input],
            generation_config=generation_config
        )

        for part in response.candidates[0].content.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                image_data = part.inline_data.data
                image = Image.open(BytesIO(image_data))

                img_io = BytesIO()
                image.save(img_io, format="PNG")
                img_io.seek(0)

                return StreamingResponse(img_io, media_type="image/png")

        raise HTTPException(status_code=500, detail="No image generated")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# import os
# import threading
# from dotenv import load_dotenv
# from collections import deque
# from urllib.parse import urlparse, parse_qs
# from datetime import datetime
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from fastapi import Response
# import json
# import time
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_chroma import Chroma
# from langchain_core.vectorstores import  InMemoryVectorStore
# from youtube_transcript_api import YouTubeTranscriptApi
# from youtube_transcript_api._errors import (
#     TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
# )

# load_dotenv()

# DATA_DIR = "./data"
# VECTOR_DIR = os.path.join(DATA_DIR, "chroma")
# os.makedirs(VECTOR_DIR, exist_ok=True)


# EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# LLM = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=1,
#     max_tokens=2048
# )

# MAX_MEMORY_TURNS = 6

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_headers=["*"],
#     allow_methods=["*"]
# )

# # ---------------- STATE ----------------
# VECTOR_STORES = {}          # video_id -> FAISS
# INDEX_STATUS = {}           # video_id -> {"status": str, "chunk_count": int}
# LOCKS = {}                  # video_id -> Lock
# MEMORY_POOL = {}            # session_video -> deque

# # Model Schemas
# class IngestURL(BaseModel):
#     url: str

# class Ask(BaseModel):
#     video_id: str
#     question: str
#     session_id: str | None = None

# # Helpers
# def extract_video_id(url: str) -> str | None:
#     parsed = urlparse(url)
#     if "youtube.com" in parsed.netloc:
#         return parse_qs(parsed.query).get("v", [None])[0]
#     if "youtu.be" in parsed.netloc:
#         return parsed.path.lstrip("/")
#     return None


# def fetch_transcript(video_id: str):
#     try:
#         return YouTubeTranscriptApi().fetch(video_id, languages=["en", "hi"])
#     except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
#         raise HTTPException(404, "Transcript unavailable")


# def get_store_path(video_id: str):
#     return os.path.join(VECTOR_DIR, video_id)

# def get_store(video_id: str):
#     return Chroma(
#         collection_name=f"video_{video_id}",
#         embedding_function=EMBEDDINGS,
#         persist_directory=get_store_path(video_id)
#     )
    

# def get_memory(session_id: str, video_id: str):
#     key = f"{session_id}_{video_id}"
#     if key not in MEMORY_POOL:
#         MEMORY_POOL[key] = deque(maxlen=MAX_MEMORY_TURNS)
#     return MEMORY_POOL[key]

# # def is_basic_question(question: str) -> bool:
# #     basic_keywords = ["hi", "hello", "hey", "how are you", "what's up", "good morning", "good evening", "thanks", "thank you", "bye", "goodbye", "see you", "who are you", "what can you do"]
# #     question_lower = question.lower().strip()
# #     return any(keyword in question_lower for keyword in basic_keywords) or len(question.split()) < 3

# # ---------------- INDEXING ----------------
# def index_video(video_id: str):

#     LOCKS.setdefault(video_id, threading.Lock())

#     with LOCKS[video_id]:

#         if INDEX_STATUS.get(video_id, {}).get("status") == "indexed":
#             return

#         INDEX_STATUS[video_id] = {"status": "indexing", "chunk_count": 0}

#         try:

#             transcript = fetch_transcript(video_id)

#             group_size = 10
#             grouped_texts = []
#             grouped_metadatas = []

#             for i in range(0, len(transcript), group_size):

#                 group = transcript[i:i+group_size]

#                 text = " ".join(c.text for c in group)

#                 metadata = {
#                     "video_id": video_id,
#                     "start": group[0].start,
#                     "duration": sum(c.duration for c in group)
#                 }

#                 grouped_texts.append(text)
#                 grouped_metadatas.append(metadata)

#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1600,
#                 chunk_overlap=160
#             )

#             docs = []

#             for text, meta in zip(grouped_texts, grouped_metadatas):

#                 chunk_docs = splitter.create_documents(
#                     [text],
#                     metadatas=[meta]
#                 )

#                 docs.extend(chunk_docs)

#             texts = []
#             metadatas = []
#             ids = []

#             for i, d in enumerate(docs):

#                 chunk_id = f"{video_id}_chunk_{i}"

#                 texts.append(d.page_content)

#                 meta = dict(d.metadata)
#                 meta["chunk_index"] = i

#                 metadatas.append(meta)

#                 ids.append(chunk_id)

#             store = get_store(video_id)

#             store.add_texts(
#                 texts=texts,
#                 metadatas=metadatas,
#                 ids=ids
#             )

#             VECTOR_STORES[video_id] = store

#             INDEX_STATUS[video_id] = {
#                 "status": "indexed",
#                 "chunk_count": len(texts)
#             }

#         except Exception:

#             INDEX_STATUS[video_id] = {
#                 "status": "failed",
#                 "chunk_count": 0
#             }


# # PROMPT TEMPLATE

# PROMPT = PromptTemplate(
#     template="""
# SYSTEM ROLE:
# You are a **YouTube video–grounded assistant**.
# Your primary job is to answer questions using the provided transcript context.
# You may also respond politely to basic conversational messages (e.g., greetings).

# CORE KNOWLEDGE RULES:
# - Use the provided transcript context as the main source of truth.
# - Do NOT invent facts, names, or explanations.
# - Do NOT fabricate timestamps or details not present.
# - If the transcript does not contain the answer, say:
# "I don't know based on this video."

# REASONABLE FLEXIBILITY:
# - If the user says "hi", "hello", or similar greetings, respond briefly and politely.
# - If the user asks about your role or capability, explain it simply (no technical jargon).
# - Do NOT answer factual or technical questions without transcript support.
# - Light conversational responses are allowed, but factual answers must stay grounded.

# ANSWERING BEHAVIOR:
# - Be clear, helpful, and readable.
# - When answering from the transcript:
#   - Explain only what is explicitly stated.
#   - Summarize only what appears in the content.
#   - If a "why" or "how" is not explained in the transcript, say you don't know.
# - If multiple viewpoints exist in the transcript, present them neutrally.
# - Preserve technical accuracy when applicable.

# CHAT HISTORY USAGE:
# - Chat history is for conversational continuity only.
# - Do NOT use chat history as a factual source.
# - If chat history conflicts with transcript, transcript always takes priority.

# TIMESTAMPS:
# - Use timestamps only if they are present in the transcript context.
# - Never invent timestamps.
# - If no timestamps are given, do not mention time.

# TONE & STYLE:
# - Natural, calm, and professional.
# - No emojis.
# - No hype.
# - No unnecessary apologies.
# - No speculation.
# - Short paragraphs or bullet points when useful.

# FAILURE MODE:
# If the transcript context does NOT contain the information required to answer a factual question, respond exactly with:
# "I don't know based on this video."

# --------------------
# CHAT HISTORY:
# {chat_history}

# --------------------
# TRANSCRIPT CONTEXT:
# {context}

# --------------------
# USER QUESTION:
# {question}

# --------------------
# FINAL ANSWER:
# """,
#     input_variables=["context", "question", "chat_history"]
# )

# # ---------------- STARTUP ----------------
# @app.on_event("startup")
# def load_existing_indexes():

#     for vid in os.listdir(VECTOR_DIR):

#         try:

#             VECTOR_STORES[vid] = Chroma(
#                 collection_name=f"video_{vid}",
#                 embedding_function=EMBEDDINGS,
#                 persist_directory=get_store_path(vid)
#             )

#             INDEX_STATUS[vid] = {
#                 "status": "indexed",
#                 "chunk_count": 0
#             }

#         except:
#             pass

# # ---------------- ROUTES ----------------
# @app.post("/ingest-url")
# def ingest(payload: IngestURL):
#     video_id = extract_video_id(payload.url)
#     if not video_id:
#         raise HTTPException(400, "Invalid YouTube URL")

#     if video_id not in VECTOR_STORES:
#         threading.Thread(
#             target=index_video,
#             args=(video_id,),
#             daemon=True
#         ).start()

#     return {"video_id": video_id, "status": "indexing"}


# @app.post("/ask")
# def ask(payload: Ask):
#     if payload.video_id not in VECTOR_STORES:
#         # Start indexing if not indexed
#         if payload.video_id not in INDEX_STATUS or INDEX_STATUS[payload.video_id].get("status") != "indexing":
#             threading.Thread(
#                 target=index_video,
#                 args=(payload.video_id,),
#                 daemon=True
#             ).start()
#         return {
#             "answer": "Indexing video, please wait a moment and ask again.",
#             "references": []
#         }

#     store = VECTOR_STORES[payload.video_id]
#     memory = get_memory(payload.session_id or "default", payload.video_id)

#     retriever = store.as_retriever(search_kwargs={"k": 6})
#     docs = retriever.invoke(payload.question)

#     context = "\n\n".join(d.page_content for d in docs)
#     refs = [{
#         "start": int(d.metadata["start"]),
#         "duration": d.metadata["duration"]
#     } for d in docs]

#     chain = (
#         RunnableParallel({
#             "context": lambda _: context,
#             "question": RunnablePassthrough(),
#             "chat_history": lambda _: "\n".join(memory)
#         })
#         | PROMPT
#         | LLM
#         | StrOutputParser()
#     )

#     answer = chain.invoke(payload.question)
#     memory.append(f"Q: {payload.question}\nA: {answer}")

#     # For basic questions, do not include timestamps
#     if is_basic_question(payload.question):
#         refs = []

#     return {
#         "answer": answer,
#         "references": refs
#     }


# @app.get("/status/{video_id}")
# def status(video_id: str):
#     return INDEX_STATUS.get(video_id, {"status": "not_indexed", "chunk_count": 0})


# @app.get("/health")
# def health():
#     return {
#         "videos": len(VECTOR_STORES),
#         "sessions": len(MEMORY_POOL)
#     }




# # ---------------- CAPTION GENERATION ----------------

# class CaptionRequest(BaseModel):
#     prompt: str


# class CaptionResponse(BaseModel):
#     youtube_caption: str


# caption_prompt = PromptTemplate(
#     template="""
# You are a social media content expert.

# User idea:
# {user_prompt}

# Generate a YouTube caption.

# Rules:
# - SEO optimized
# - Slightly longer
# - No hashtags inside sentence

# Return strictly in JSON format:
# {{
#     "youtube_caption": "..."
# }}
# """,
#     input_variables=["user_prompt"]
# )

# caption_parser = StrOutputParser()

# caption_model = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.6
# )

# caption_chain = caption_prompt | caption_model | caption_parser


# @app.post("/generate-caption", response_model=CaptionResponse)
# async def generate_caption(request: CaptionRequest):
#     try:
#         result = caption_chain.invoke({"user_prompt": request.prompt})

#         import json
#         parsed = json.loads(result)

#         return {
#             "youtube_caption": parsed["youtube_caption"]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # ---------------- SEMANTIC SIMILARITY SCORING ----------------

# class SimilarityRequest(BaseModel):
#     video_id: str
#     query: str
#     top_k: int = 5


# class SimilarityResult(BaseModel):
#     segment_id: int
#     similarity_score: float
#     start: int
#     duration: float
#     text: str


# @app.post("/semantic-similarity")
# def semantic_similarity(payload: SimilarityRequest):

#     if payload.video_id not in VECTOR_STORES:
#         raise HTTPException(404, "Video index not found")

#     store = VECTOR_STORES[payload.video_id]

#     # embed query
#     query_vector = EMBEDDINGS.embed_query(payload.query)

#     # FAISS similarity search with scores
#     docs_scores = store.similarity_search_with_score(
#         payload.query,
#         k=payload.top_k
#     )

#     results = []

#     for i, (doc, score) in enumerate(docs_scores):
#         results.append({
#             "segment_id": i,
#             "similarity_score": float(score),
#             "start": int(doc.metadata.get("start", 0)),
#             "duration": doc.metadata.get("duration", 0),
#             "text": doc.page_content
#         })

#     return {
#         "video_id": payload.video_id,
#         "query": payload.query,
#         "results": results
#     }


# # ---------------- THUMBNAIL TEXT SUGGESTIONS ----------------

# class ThumbnailRequest(BaseModel):
#     prompt: str


# class ThumbnailResponse(BaseModel):
#     suggestions: list[str]


# thumbnail_prompt = PromptTemplate(
#     template="""
# You are a YouTube growth strategist.

# User topic:
# {user_prompt}

# Generate short thumbnail text phrases.

# Rules:
# - 2 to 6 words per phrase
# - Attention grabbing
# - High curiosity
# - Suitable for YouTube thumbnails
# - No emojis
# - No hashtags

# Return strictly JSON:

# {{
#   "suggestions": [
#     "...",
#     "...",
#     "...",
#     "...",
#     "..."
#   ]
# }}
# """,
#     input_variables=["user_prompt"]
# )

# thumbnail_model = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.7
# )

# thumbnail_parser = StrOutputParser()

# thumbnail_chain = thumbnail_prompt | thumbnail_model | thumbnail_parser


# @app.post("/thumbnail-text", response_model=ThumbnailResponse)
# async def generate_thumbnail_text(request: ThumbnailRequest):
#     try:
#         result = thumbnail_chain.invoke({"user_prompt": request.prompt})

#         import json
#         parsed = json.loads(result)

#         return {
#             "suggestions": parsed["suggestions"]
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
# # ---------------- CONTENT SUMMARIZATION ----------------

# class SummaryRequest(BaseModel):
#     video_id: str
#     max_segments: int = 8


# class SummaryResponse(BaseModel):
#     summary: str


# summary_prompt = PromptTemplate(
#     template="""
# You are a precise summarization system.

# Task:
# Summarize the provided transcript segments.

# Rules:
# - Extract only key ideas.
# - Do not invent information.
# - Write concise and clear paragraphs.
# - Focus on the main concepts discussed.

# Transcript Segments:
# {context}

# Return only the summary text.
# """,
#     input_variables=["context"]
# )

# summary_model = ChatGroq(
#     model="openai/gpt-oss-120b",
#     api_key=os.getenv("GROQ_API_KEY"),
#     temperature=0.3
# )

# summary_parser = StrOutputParser()

# summary_chain = summary_prompt | summary_model | summary_parser


# @app.post("/summarize", response_model=SummaryResponse)
# def summarize_video(payload: SummaryRequest):

#     if payload.video_id not in VECTOR_STORES:
#         raise HTTPException(404, "Video index not found")

#     store = VECTOR_STORES[payload.video_id]

#     docs = store.similarity_search(
#         "main topics of the video",
#         k=payload.max_segments
#     )

#     context = "\n\n".join(d.page_content for d in docs)

#     summary = summary_chain.invoke({"context": context})

#     return {
#         "summary": summary
#     }

# class TranscriptSchema(BaseModel):
#     video_id: str
#     start_time: float
#     end_time: float
#     text: str
#     embedding_id: str
#     createdAt: datetime
#     updatedAt: datetime
    
# SCHEMA_DIR = "./data/transcripts"
# os.makedirs(SCHEMA_DIR, exist_ok=True)


# @app.post("/transcript-schema")
# def transcript_schema(payload: IngestURL):

#     video_id = extract_video_id(payload.url)
#     if not video_id:
#         raise HTTPException(400, "Invalid YouTube URL")

#     transcript = fetch_transcript(video_id)

#     group_size = 10
#     results = []

#     for i in range(0, len(transcript), group_size):

#         group = transcript[i:i+group_size]

#         start_time = group[0].start
#         end_time = group[-1].start + group[-1].duration

#         text = " ".join(c.text for c in group)

#         embedding_id = f"{video_id}_vec_{i//group_size}"

#         results.append({
#             "video_id": video_id,
#             "start_time": start_time,
#             "end_time": end_time,
#             "text": text,
#             "embedding_id": embedding_id,
#             "createdAt": datetime.utcnow().isoformat(),
#             "updatedAt": datetime.utcnow().isoformat()
#         })

#     file_path = os.path.join(SCHEMA_DIR, f"{video_id}.json")

#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     return {
#         "video_id": video_id,
#         "saved_to": file_path,
#         "segments": results
#     }




# @app.get("/vector-ids/{video_id}")
# def get_vector_ids(video_id: str):

#     if video_id not in VECTOR_STORES:
#         raise HTTPException(404, "Video not indexed")

#     store = VECTOR_STORES[video_id]

#     data = store.get()

#     return {
#         "video_id": video_id,
#         "vector_ids": data.get("ids", [])
#     }

# @app.post("/search-video")
# def search_video(payload: SimilarityRequest):

#     if payload.video_id not in VECTOR_STORES:
#         raise HTTPException(404, "Video not indexed")

#     store = VECTOR_STORES[payload.video_id]

#     docs = store.similarity_search(payload.query, k=payload.top_k)

#     results = []

#     for d in docs:
#         results.append({
#             "embedding_id": d.metadata.get("embedding_id"),
#             "start_time": d.metadata.get("start"),
#             "duration": d.metadata.get("duration"),
#             "text": d.page_content
#         })

#     return {
#         "video_id": payload.video_id,
#         "query": payload.query,
#         "results": results
#     }


# import os
# import base64
# from io import BytesIO
# from PIL import Image
# from google import genai
# from google.genai import types



# client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# # ── Prompt Builder ────────────────────────────────────────────────────────────

# def build_prompt(
#     topic: str,
#     style: str = "bold and eye-catching",
#     text_overlay: str | None = None,
#     extra: str | None = None,
# ) -> str:
#     parts = [
#         "YouTube thumbnail, 16:9 aspect ratio, 1280x720 pixels.",
#         f"Topic: {topic}.",
#         f"Style: {style}, high contrast, vibrant colors, professional quality.",
#         "No letterboxing. Fill the entire frame.",
#     ]
#     if text_overlay:
#         parts.append(f'Bold text overlay: "{text_overlay}". Large readable font.')
#     if extra:
#         parts.append(extra)
#     return " ".join(parts)

# # ── Core Generator ────────────────────────────────────────────────────────────

# def generate_thumbnail(
#     topic: str,
#     style: str = "bold and eye-catching",
#     text_overlay: str | None = None,
#     extra_instructions: str | None = None,
#     output_path: str | None = None,
# ) -> bytes:
#     prompt = build_prompt(topic, style, text_overlay, extra_instructions)

#     response = client.models.generate_content(
#         model="gemini-3.1-flash-image-preview",  # ✅ only free-tier model
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_modalities=["IMAGE", "TEXT"],
#         ),
#     )

#     img_bytes = None
#     for part in response.candidates[0].content.parts:
#         if part.inline_data:
#             img_bytes = part.inline_data.data
#             break

#     if not img_bytes:
#         raise ValueError("No image returned by the model.")

#     if output_path:
#         img = Image.open(BytesIO(img_bytes))
#         img.save(output_path, format="JPEG", quality=95)

#     return img_bytes
# # def generate_thumbnail(
# #     topic: str,
# #     style: str = "bold and eye-catching",
# #     text_overlay: str | None = None,
# #     extra_instructions: str | None = None,
# #     output_path: str | None = None,
# # ) -> bytes:
# #     prompt = build_prompt(topic, style, text_overlay, extra_instructions)

# #     response = client.models.generate_content(
# #         model="gemini-3.1-flash-image-preview",
# #         contents=prompt,
# #         config=types.GenerateContentConfig(
# #             response_modalities=["IMAGE", "TEXT"],
# #         ),
# #     )

# #     img_bytes = None
# #     for part in response.candidates[0].content.parts:
# #         if part.inline_data:
# #             img_bytes = part.inline_data.data
# #             break

# #     if not img_bytes:
# #         raise ValueError("No image returned by the model.")

# #     if output_path:
# #         img = Image.open(BytesIO(img_bytes))
# #         img.save(output_path, format="JPEG", quality=95)
# #         print(f"Saved → {output_path}")

# #     return img_bytes

# # ── Rate Limit Safe Wrapper ───────────────────────────────────────────────────

# def generate_thumbnail_safe(retries: int = 3, wait: int = 10, **kwargs) -> bytes:
#     for attempt in range(retries):
#         try:
#             return generate_thumbnail(**kwargs)
#         except Exception as e:
#             if "429" in str(e) and attempt < retries - 1:
#                 print(f"Rate limited. Retrying in {wait}s...")
#                 time.sleep(wait)
#             else:
#                 raise

# # ── Helpers ───────────────────────────────────────────────────────────────────

# def bytes_to_base64(img_bytes: bytes) -> str:
#     return base64.b64encode(img_bytes).decode("utf-8")

# # ── FastAPI App ───────────────────────────────────────────────────────────────

# class ThumbnailRequest(BaseModel):
#     topic: str
#     style: str = "bold and eye-catching"
#     text_overlay: str | None = None
#     extra_instructions: str | None = None

# class ThumbnailResponse(BaseModel):
#     image: str  # base64-encoded JPEG

# @app.post("/generate-thumbnail", response_model=ThumbnailResponse)
# def generate(req: ThumbnailRequest):
#     try:
#         raw = generate_thumbnail_safe(
#             topic=req.topic,
#             style=req.style,
#             text_overlay=req.text_overlay,
#             extra_instructions=req.extra_instructions,
#         )
#         return ThumbnailResponse(image=bytes_to_base64(raw))
#     except Exception as e:
#         raise HTTPException(500, str(e))

# @app.get("/generate-thumbnail/preview")
# def preview(topic: str, text_overlay: str | None = None):
#     """Returns the image directly — open in browser to preview."""
#     raw = generate_thumbnail_safe(topic=topic, text_overlay=text_overlay)
#     return Response(content=raw, media_type="image/jpeg")
