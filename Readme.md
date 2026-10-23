# RAG-Based AI Teaching Assistant

A Streamlit app that turns a lecture video into a question-answering assistant. Upload a video, and the app transcribes it with Whisper, builds embeddings with Ollama, and lets you ask questions in plain language. Answers come with the source chunks and their timestamps so you can jump to the right part of the video.

Everything runs locally: no API keys and no cloud services.

## How it works

1. **Video to audio**: `ffmpeg` extracts an `.mp3` from the uploaded video.
2. **Transcription**: OpenAI Whisper (`base` model) transcribes the audio into timestamped segments, saved as JSON.
3. **Embeddings**: each segment is embedded with `bge-m3` through Ollama and stored in `embeddings/embeddings.joblib`.
4. **Question answering**: your question is embedded, the most similar segments are retrieved by cosine similarity, and `llama3.2` answers using only those segments.

## Requirements

- **Python 3.12 or newer** (the code uses an f-string syntax that older versions reject)
- **[ffmpeg](https://ffmpeg.org/)** installed and available on your PATH
- **[Ollama](https://ollama.com/)** installed and running
- Enough RAM/CPU to run Whisper and `llama3.2` locally (a GPU helps but is not required)

## Setup

### 1. Install ffmpeg

```bash
# Windows (Chocolatey)
choco install ffmpeg

# Windows (Scoop)
scoop install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg
```

Check it works: `ffmpeg -version`

### 2. Install Ollama and pull the models

Install Ollama from <https://ollama.com/>, make sure it is running, then:

```bash
ollama pull bge-m3
ollama pull llama3.2
```

### 3. Clone the repo and create a virtual environment

```bash
git clone https://github.com/MisterKinetic/rag-based-ai.git
cd rag-based-ai

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

The first install pulls in PyTorch through `openai-whisper`, so it can take a few minutes. Whisper also downloads its `base` model the first time you transcribe something.

### 5. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser (usually <http://localhost:8501>).

## Usage

1. Pick the lecture's language in the sidebar.
2. Upload a video (MP4, AVI, MKV, MOV or WEBM).
3. Click **Process Lecture** and wait for transcription and embedding to finish.
4. Type a question and click **Ask**.
5. Read the answer and check the listed sources and timestamps.

## Configuration

Settings can be changed with environment variables or a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Address of the Ollama server |
| `EMBED_MODEL` | `bge-m3` | Ollama model used for embeddings |
| `LLM_MODEL` | `llama3.2` | Ollama model used to generate answers |
| `EMBED_BATCH_SIZE` | `32` | Number of texts embedded per request |

Example `.env`:

```
LLM_MODEL=llama3.2
EMBED_MODEL=bge-m3
```

Do not commit your `.env` file. It is already listed in `.gitignore`.

## Project structure

```
app.py            Streamlit user interface
rag_utils.py      Video/audio processing, transcription, embeddings, retrieval, LLM calls
requirements.txt  Python dependencies
```

Folders created at runtime (all git-ignored): `videos/`, `audios/`, `jsons/`, `embeddings/`, `logs/`.

## Known limitations

- Embeddings are rebuilt from **all** transcripts in `jsons/`, so older lectures stay in the knowledge base when you process a new video. Delete the contents of `jsons/` and `embeddings/` to start fresh.
- If you upload a different video with the same filename as an earlier one, the old audio file is reused. Delete the matching file in `audios/` or rename the video.
- The "Reset session" button removes the uploaded video but not the generated audio, transcripts or embeddings.
- Unrelated questions may still receive an answer built from loosely related chunks, because relevance is not filtered by a similarity threshold.
- Answers on a CPU-only machine can be slow and may hit the 60-second LLM timeout.

## Troubleshooting

- **`ffmpeg` not found**: ffmpeg is not installed or not on your PATH. Reinstall it and open a new terminal.
- **Connection refused on port 11434**: Ollama is not running. Start the Ollama app or run `ollama serve`.
- **Model not found**: run `ollama pull bge-m3` and `ollama pull llama3.2`.
- **Syntax error in `rag_utils.py` on startup**: you are on Python 3.11 or older. Upgrade to Python 3.12+.

## Credits

Transcription uses [OpenAI Whisper](https://github.com/openai/whisper) (MIT License). Embeddings and answers are served locally by [Ollama](https://ollama.com/).

This project is based on an existing open-source RAG lecture assistant, adapted and fixed to run end to end. Add the original author and repository link here, and check the original license before redistributing.
