# ⚖️ LegalMind

**AI-Powered Legal Document Assistant — Local, Private, Accurate**

> Upload any legal PDF. Ask questions in plain English. Get answers grounded strictly in your document — no hallucinations, no external APIs, no data leaving your machine.

---

## What It Does

LegalMind is a **Retrieval-Augmented Generation (RAG)** backend for legal documents. Upload a contract, act, agreement, or policy — then interrogate it like a lawyer would, in plain English.

It operates in two modes:

| Mode | Behaviour |
|---|---|
| `explain` | Breaks down a legal clause or term into simple language |
| `scenario` | Applies the document's legal text to a real-world situation you describe |

Every answer is grounded strictly in the uploaded document. If the answer isn't there, LegalMind says so — it never fills gaps with invented law.

---

## How It Works

```
User uploads PDF
      ↓
Text extracted                   (PyPDF2)
      ↓
Split into overlapping chunks    (custom text_splitter — 500 words, 50-word overlap)
      ↓
Chunks encoded into vectors      (all-MiniLM-L6-v2)
      ↓
Vectors stored per-session       (ChromaDB — isolated collections)
      ↓
User asks a question
      ↓
Question embedded → top-3 chunks retrieved by cosine similarity
      ↓
Chunks + question sent to Mistral (via Ollama)
      ↓
Grounded plain-English answer returned
```

---

## Project Structure

```
├── app.py                  # FastAPI entrypoint — all routes
├── backend/
│   ├── pdf_reader.py       # Extracts raw text from uploaded PDFs
│   ├── text_splitter.py    # Splits text into overlapping word-count chunks
│   ├── embeddings.py       # Encodes chunks into sentence vectors
│   ├── model_loader.py     # Singleton loader for the SentenceTransformer model
│   ├── vector_store.py     # ChromaDB store + top-k retrieval (per session)
│   ├── search.py           # Cosine similarity search helper
│   └── llm_explainer.py    # Prompts Mistral via Ollama for final answers
├── evaluation/
│   ├── eval.py             # DeepEval RAG evaluation pipeline
│   └── data.json           # Ground-truth Q&A pairs
├── uploads/                # Uploaded PDFs (namespaced by session_id)
├── chroma_db/              # ChromaDB persistent storage
└── sessions.json           # Active session state (survives restarts)
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.ai/) installed and running locally

```bash
ollama pull mistral
```

### Install Dependencies

```bash
pip install fastapi uvicorn python-multipart \
            PyPDF2 sentence-transformers \
            chromadb scikit-learn ollama \
            deepeval numpy
```

### Run the Server

```bash
uvicorn app:app --reload
```

API available at `http://localhost:8000`.

---

## API Reference

### `GET /health`
Confirm the server is running.

**Response**
```json
{ "status": "ok" }
```

---

### `POST /session/create`
Creates an isolated session. Returns a `session_id` and sets an `httponly` cookie.

**Response**
```json
{ "message": "Session created", "session_id": "f47ac10b-58cc-..." }
```

---

### `POST /upload`
Upload a PDF. The document is extracted, chunked, embedded, and indexed immediately.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | PDF file | The legal document to upload |

**Response**
```json
{ "message": "PDF uploaded and processed.", "session_id": "f47ac10b-58cc-..." }
```

---

### `POST /ask_question`
Ask a natural-language question about the uploaded document.

**Request** — `application/json`

| Field | Type | Required | Description |
|---|---|---|---|
| `session_id` | `str` | ✅ | Your session UUID |
| `question` | `str` | ✅ | Natural-language question |
| `mode` | `str` | optional | `explain` (default) or `scenario` |

**Response**
```json
{
  "session": "f47ac10b-58cc-...",
  "answer": "Either party may terminate the contract with 30 days written notice...",
  "mode": "explain"
}
```

---

### `DELETE /session/clear`
Deletes the session, removes the uploaded PDF, and wipes the ChromaDB collection.

**Response**
```json
{ "message": "Session cleared." }
```

---

## Configuration

All tuneable parameters are in one place:

| Parameter | File | Default | Notes |
|---|---|---|---|
| Chunk size | `text_splitter.py` | `500` words | Larger chunks = more context per retrieval |
| Overlap | `text_splitter.py` | `50` words | Prevents context loss at boundaries |
| Top-k retrieved chunks | `vector_store.py` | `3` | Increase to `5` to reduce faithfulness gaps |
| Embedding model | `model_loader.py` | `all-MiniLM-L6-v2` | Fast, local, no API key needed |
| LLM model | `llm_explainer.py` | `mistral` | Any Ollama-compatible model works |

---

## Evaluation

LegalMind ships with a RAG evaluation pipeline using [DeepEval](https://github.com/confident-ai/deepeval). Mistral itself acts as the judge — no external API keys needed.

### Metrics (passing threshold: 0.7)

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer stay within the retrieved chunks — no invented facts? |
| **Answer Relevancy** | Does the answer directly address what was asked? |
| **Contextual Precision** | Are the most relevant chunks ranked highest during retrieval? |

### Run

```bash
# Run as a module so backend/ imports resolve correctly
python -m evaluation.eval
```

### Latest Results

Evaluated on a PIL challenging Section 166(3) of the Motor Vehicles Act, 1988 — 12 questions.

| # | Question | Faithfulness | Answer Relevancy | Contextual Precision |
|---|---|:---:|:---:|:---:|
| 1 | Constitutional provision for PIL | 0.75 | 0.43 | 0.83 |
| 2 | Which MV Act provision is challenged | 0.60 | 1.00 | 1.00 |
| 3 | Limitation period introduced by amendment | 0.50 | — | 0.72 |
| 4 | Why is the six-month limit harmful | 0.67 | 0.83 | 1.00 |
| 5 | Pre-2019 limitation rule | 0.50 | 0.60 | 1.00 |
| 6 | Who is the petitioner | 1.00 | 0.50 | 0.44 |
| 7 | Government authority as respondent | 0.50 | 0.67 | 0.50 |
| 8 | Section 159 police responsibility | 0.50 | 0.50 | 1.00 |
| 9 | Fundamental rights violated | 0.50 | 0.14 | 0.44 |
| 10 | Interim relief requested | 0.67 | 0.50 | 0.83 |
| 11 | Why is the limitation arbitrary | 0.60 | 1.00 | 1.00 |
| 12 | Type of legislation the MV Act is | 0.67 | 1.00 | 1.00 |
| | **Average** | **0.621 ❌** | **0.681 ⚠️** | **0.814 ✅** |

### Interpreting the Results

**Contextual Precision — 0.814 ✅** — Retrieval is working well; the right chunks surface at the top for most queries. Q6 and Q9 score low (0.44) because the answer is spread across non-adjacent chunks — increase overlap in `text_splitter.py` to fix.

**Answer Relevancy — 0.681 ⚠️** — Just under threshold. The LLM sometimes answers around the question (Q9: 0.14, Q1: 0.43). Add a few-shot example to the prompt in `llm_explainer.py` that leads with a direct one-sentence answer.

**Faithfulness — 0.621 ❌** — Below threshold. The model occasionally introduces facts not in the retrieved chunks when context is sparse. Fix: tighten the system prompt with an explicit constraint (`Answer only from the text below. If it's not there, say so.`) and increase `top_k` from `3` → `5` in `vector_store.py`.

---

## Architecture Notes

**Per-session ChromaDB collections** — Each session gets its own `session_<uuid>` collection. Documents from different users are never mixed. Deleting a session wipes the collection.

**Overlapping chunking** — 50-word overlap carries context across chunk boundaries, which matters for legal text where a single clause can span several sentences.

**Top-k context merging** — The top 3 retrieved chunks are joined with `---` separators and sent together to the LLM. More context means fewer gaps to hallucinate across.

**Fully local stack** — Embedding model, vector DB, and LLM all run on your machine. No API keys, no network calls, no telemetry.

---

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| PDF Parsing | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| Embeddings | [SentenceTransformers](https://www.sbert.net/) — `all-MiniLM-L6-v2` |
| Vector Store | [ChromaDB](https://www.trychroma.com/) |
| LLM | [Mistral](https://mistral.ai/) via [Ollama](https://ollama.ai/) |
| Evaluation | [DeepEval](https://github.com/confident-ai/deepeval) |
| Session Storage | JSON file + HTTP cookies |

---

## Example Queries

```
"Does this contract allow subletting?"
→ Searches the lease agreement for subletting clauses, explains in plain English.

"My employer hasn't paid me in 3 weeks — what does the labour act say?"
→ Scenario mode: applies the uploaded act to the specific situation.

"Summarise the liability clause."
→ Explain mode: condenses dense legal language into one clear paragraph.

"Is there a penalty for late payment?"
→ Locates and explains any penalty or interest provisions in the document.
```

---

## Privacy & Safety

- All processing is **local** — documents never leave your machine
- Sessions are **fully isolated** at the vector store level
- The LLM is prompted to answer **only from the provided document text** and will explicitly say when information is absent rather than guess
- Uploaded files are namespaced by `session_id` and deleted on session clear

---

## License

MIT — free to use, modify, and distribute.

---

*FastAPI · ChromaDB · SentenceTransformers · Mistral · Ollama · DeepEval*
