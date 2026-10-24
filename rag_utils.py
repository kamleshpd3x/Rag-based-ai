import os
import subprocess
import json
import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Optional: load environment variables from .env (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configuration – can be overridden via environment variables
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create required directories if they do not exist."""
    for d in ["audios", "jsons", "embeddings", "logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def hash_text(text: str) -> str:
    """Return a short SHA256 hash of the given text.

    Used to avoid re‑embedding identical chunks.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Video / Audio processing
# ---------------------------------------------------------------------------

def convert_video_to_audio(video_path: str) -> str:
    """Extract audio from a video file using ffmpeg.

    Returns the path to the produced ``.mp3`` file.
    """
    ensure_dirs()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    # Build a deterministic audio filename based on the video name
    audio_name = f"{video_path.stem}.mp3"
    audio_path = Path("audios") / audio_name
    # Skip conversion only if the audio is newer than the video (a replaced video must be re-extracted)
    if audio_path.is_file() and audio_path.stat().st_mtime >= video_path.stat().st_mtime:
        logger.info("Audio already exists: %s", audio_path)
        return str(audio_path)
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i",
        str(video_path),
        str(audio_path),
    ]
    logger.info("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    logger.info("Created audio: %s", audio_path)
    return str(audio_path)

# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _whisper_model():
    """Load Whisper once per process instead of on every lecture."""
    import whisper  # Imported lazily to avoid heavy import when not needed

    return whisper.load_model("base")


def transcribe_audio(
    audio_path: str, language: str = "en", task: str = "transcribe"
) -> str:
    """Transcribe an audio file with Whisper and write a JSON file.

    The JSON contains a list of ``chunks`` – each with ``number``, ``title``,
    ``start``, ``end`` and ``text`` – plus a ``text`` field with the full
    transcription.
    """
    ensure_dirs()
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # Derive a simple *number* and *title* from the filename (e.g. "01 Lecture.mp3")
    number = audio_path.stem.split(" ")[0]
    title = audio_path.stem
    model = _whisper_model()
    logger.info("Transcribing %s with Whisper (language=%s, task=%s)", audio_path, language, task)
    result = model.transcribe(str(audio_path), language=language, task=task)
    chunks = []
    for segment in result.get("segments", []):
        chunks.append(
            {
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
            }
        )
    json_obj = {"chunks": chunks, "text": result.get("text", "")}
    json_path = Path("jsons") / f"{audio_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, indent=2)
    logger.info("Wrote transcription JSON: %s", json_path)
    return str(json_path)

# ---------------------------------------------------------------------------
# Embedding utilities (Ollama)
# ---------------------------------------------------------------------------

def create_embedding(text_list: List[str]) -> List[Any]:
    """Call Ollama embed endpoint, sending texts in batches.

    Returns a list of embeddings matching ``text_list`` (same order).
    """
    if not text_list:
        return []
    embeddings: List[Any] = []
    url = f"{OLLAMA_URL}/api/embed"
    for start in range(0, len(text_list), EMBED_BATCH_SIZE):
        batch = text_list[start:start + EMBED_BATCH_SIZE]
        payload = {"model": EMBED_MODEL, "input": batch}
        try:
            resp = requests.post(url, json=payload, timeout=max(30, 10 * len(batch)))
            resp.raise_for_status()
            vectors = resp.json().get("embeddings", [])
        except Exception as e:
            logger.error("Embedding request failed: %s", e)
            raise
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(batch)} texts"
            )
        embeddings.extend(vectors)
        logger.info(
            "Embedded %d/%d texts",
            min(start + len(batch), len(text_list)),
            len(text_list),
        )
    return embeddings

def _load_embedding_cache(path: Path) -> Dict[str, Any]:
    """Return ``{text_hash: embedding}`` from an existing embeddings store.

    Unreadable or malformed stores are ignored (logged) so the next build
    simply recomputes them.
    """
    if not path.is_file():
        return {}
    try:
        prev = joblib.load(path)
        cache: Dict[str, Any] = {}
        for text, emb in zip(prev["text"], prev["embedding"]):
            if emb is None or (isinstance(emb, float) and np.isnan(emb)):
                continue
            cache[hash_text(str(text))] = emb
        logger.info("Reusing %d cached embeddings from %s", len(cache), path)
        return cache
    except Exception as e:
        logger.warning("Could not reuse existing embeddings (%s); rebuilding.", e)
        return {}


def build_embeddings(json_dir: str = "jsons") -> str:
    """Read every JSON in *json_dir*, embed new/changed chunks, store a DataFrame.

    Vectors whose text hash already exists in ``embeddings/embeddings.joblib``
    are reused, so re-processing only pays for new or changed content. The
    resulting DataFrame is saved to ``embeddings/embeddings.joblib`` and also
    returned as a path string.
    """
    ensure_dirs()
    json_dir_path = Path(json_dir)
    if not json_dir_path.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir_path}")
    embed_path = Path("embeddings") / "embeddings.joblib"

    # The JSON files are the source of truth; the store only supplies vectors.
    all_chunks: List[Dict[str, Any]] = []
    for json_file in sorted(json_dir_path.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            content = json.load(f)
        all_chunks.extend(content.get("chunks", []))

    embedding_cache = _load_embedding_cache(embed_path)
    # Texts that are not cached yet, deduplicated while keeping order.
    missing: Dict[str, str] = {}
    for chunk in all_chunks:
        txt_hash = hash_text(chunk["text"])
        if txt_hash not in embedding_cache and txt_hash not in missing:
            missing[txt_hash] = chunk["text"]
    if missing:
        logger.info(
            "Embedding %d new texts (%d reused from cache)",
            len(missing),
            len(embedding_cache),
        )
        for txt_hash, emb in zip(missing, create_embedding(list(missing.values()))):
            embedding_cache[txt_hash] = emb
    else:
        logger.info(
            "All %d chunks already embedded; skipping embedding", len(all_chunks)
        )

    records = []
    for chunk in all_chunks:
        chunk_copy = chunk.copy()
        chunk_copy["embedding"] = embedding_cache[hash_text(chunk["text"])]
        records.append(chunk_copy)
    df = pd.DataFrame.from_records(records)
    joblib.dump(df, embed_path)
    logger.info("Saved embeddings to %s (rows=%d)", embed_path, len(df))
    return str(embed_path)

def load_embeddings(path: str = "embeddings/embeddings.joblib") -> pd.DataFrame:
    return joblib.load(path)

# ---------------------------------------------------------------------------
# LLM inference (Ollama)
# ---------------------------------------------------------------------------

def inference(prompt: str) -> str:
    """Generate a response from the configured LLM via Ollama.
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except Exception as e:
        logger.error("LLM request failed: %s", e)
        raise

# ---------------------------------------------------------------------------
# Query pipeline
# ---------------------------------------------------------------------------

def answer_question(
    question: str, top_k: int = 3, embeddings_path: str = "embeddings/embeddings.joblib"
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return a natural‑language answer and the source chunks used.

    1. Embed the question.
    2. Compute cosine similarity against stored chunk embeddings.
    3. Pick the *top_k* most similar chunks.
    4. Build a concise prompt for the LLM and return its response.
    """
    df = load_embeddings(embeddings_path)
    if df.empty:
        raise RuntimeError("Embedding store is empty – run the preprocessing step first.")
    q_emb = create_embedding([question])[0]
    similarities = cosine_similarity(np.vstack(df["embedding"].values), [q_emb]).flatten()
    top_idx = similarities.argsort()[::-1][:top_k]
    top_chunks = df.iloc[top_idx]
    # If similarity scores are all zero, we treat it as no relevant content
    if similarities[top_idx].max() == 0:
        return ("I couldn't find relevant information in the lecture material.", [])
    # Build a short prompt – include title, timestamps and text for each chunk
    chunk_texts = []
    for _, row in top_chunks.iterrows():
        chunk_texts.append(
            f"Video: {row['title']} (Chunk {row['number']}) [{row['start']:.1f}s - {row['end']:.1f}s]\nText: {row['text']}"
        )
    prompt = (
        "You are a helpful AI assistant for a lecture series. Use only the information "
        "provided below to answer the user's question. If the question is unrelated, "
        "politely state that you can only answer questions about the lecture content.\n\n"
        f"Relevant chunks:\n{'\n'.join(chunk_texts)}\n\nQuestion: {question}\n\nAnswer:"  
    )
    answer = inference(prompt)
    # Return answer and raw source chunks for display
    sources = top_chunks.to_dict(orient="records")
    return answer.strip(), sources

# ---------------------------------------------------------------------------
# Convenience wrappers used by Streamlit UI
# ---------------------------------------------------------------------------

def process_video(video_path: str, language: str = "en") -> str:
    """Full pipeline: video → audio → transcription → embedding.

    Returns the path to the embeddings file.
    """
    audio = convert_video_to_audio(video_path)
    trans_json = transcribe_audio(audio, language=language)
    with open(trans_json, encoding="utf-8") as f:
        if not json.load(f)["chunks"]:
            raise ValueError("No speech found in the video's audio track.")
    # Re‑build the whole embedding store – cheap for a demo with a few files
    embed_path = build_embeddings()
    return embed_path

"""End of rag_utils module"""
