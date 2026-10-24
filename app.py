import os
import shutil
from pathlib import Path
import streamlit as st

# Import our utility module
import rag_utils

# ---------------------------------------------------------------------------
# Helper to store uploaded videos
# ---------------------------------------------------------------------------
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit uploaded file to the local *videos* directory.
    Returns the absolute path to the saved file.
    """
    # Use only the base name so a crafted upload filename cannot escape VIDEO_DIR
    file_path = VIDEO_DIR / Path(uploaded_file.name).name
    # Overwrite if the same name exists
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Lecture Assistant", layout="centered")
st.title("📚 RAG‑Based AI Teaching Assistant")
st.write(
    "Upload a lecture video, let the system transcribe it with Whisper, build embeddings, "
    "and ask natural‑language questions. Answers include timestamps so you can jump "
    "directly to the relevant video segment."
)

# Sidebar – language selection (default English) and optional settings
with st.sidebar:
    st.header("Settings")
    language = st.selectbox(
        "Transcription language",
        ["en", "hi", "es", "fr", "de", "zh", "ja"],
        index=0,
        help="Select the language of the lecture for Whisper. "
    )
    top_k = st.slider("Number of retrieved chunks", min_value=1, max_value=5, value=3)

# Session state flags
if "processed" not in st.session_state:
    st.session_state.processed = False
if "emb_path" not in st.session_state:
    st.session_state.emb_path = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None

# ---------------------------------------------------------------
# Step 1 – Upload video
# ---------------------------------------------------------------
uploaded = st.file_uploader(
    "Upload a lecture video (MP4, AVI, MKV …)",
    type=["mp4", "avi", "mkv", "mov", "webm"],
    help="The file is saved locally and later processed."
)
if uploaded:
    st.success(f"Uploaded: {uploaded.name}")
    # The file uploader keeps its value across reruns, so only save/reset when a
    # *new* file arrives. Doing it unconditionally wiped the "processed" flag on
    # every interaction (e.g. typing a question), hiding the Q&A panel.
    # Compare file_id, not name: a different video re-uploaded as "video.mp4" must still be saved.
    if st.session_state.get("uploaded_id") != uploaded.file_id:
        video_path = save_uploaded_file(uploaded)
        st.session_state.video_path = video_path
        st.session_state.uploaded_id = uploaded.file_id
        # Reset previous processing if a new video is uploaded
        st.session_state.processed = False
        st.session_state.emb_path = None

# ---------------------------------------------------------------
# Step 2 – Process lecture (transcribe + embed)
# ---------------------------------------------------------------
if st.session_state.video_path:
    if st.button("▶️ Process Lecture"):
        with st.spinner("Converting video, transcribing, and building embeddings…"):
            try:
                # This function runs the full pipeline and returns the embeddings path
                emb_path = rag_utils.process_video(st.session_state.video_path, language=language)
                st.session_state.emb_path = emb_path
                st.session_state.processed = True
                st.success("✅ Lecture processed! You can now ask questions.")
            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.session_state.processed = False

# ---------------------------------------------------------------
# Step 3 – Question answering UI (appears after processing)
# ---------------------------------------------------------------
if st.session_state.processed:
    st.subheader("💬 Ask a question about the lecture")
    question = st.text_input("Your question", "")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving relevant chunks and generating answer…"):
                try:
                    answer, sources = rag_utils.answer_question(
                        question,
                        top_k=top_k,
                        embeddings_path=st.session_state.emb_path,
                    )
                    st.markdown(f"**Answer:**\n{answer}")
                    if sources:
                        st.markdown("---\n**Sources**")
                        for src in sources:
                            ts = f"{src['start']:.1f}s – {src['end']:.1f}s"
                            st.markdown(
                                f"* **{src['title']}** (Chunk {src['number']}, {ts})\n  > {src['text'].strip()}"
                            )
                    else:
                        st.info("No relevant content was found in the lecture material.")
                except Exception as e:
                    st.error(f"Error during inference: {e}")

# ---------------------------------------------------------------
# Clean‑up helper (optional)
# ---------------------------------------------------------------
if st.sidebar.button("🗑️ Reset session"):
    # Remove uploaded files and reset state – useful during demo iterations
    if st.session_state.video_path:
        try:
            os.remove(st.session_state.video_path)
        except Exception:
            pass
    st.session_state.clear()
    st.rerun()
