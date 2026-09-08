# KasRAG

A minimal Retrieval-Augmented Generation system in plain Python — no LangChain, no vector
database service, and **no API keys**. Both models run locally through
[Ollama](https://ollama.com), so the whole thing works offline once the models are pulled.

Ask a question, the app finds the most relevant facts from a local text file by cosine
similarity, pastes them into the prompt, and a small Llama model answers from that context
alone. The Streamlit UI shows the retrieved chunks and their similarity scores next to every
answer, so you can see what the response was actually built from.

## Where this came from

I built this by following **[Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag)**
by [ngxson](https://huggingface.co/ngxson) — a walkthrough of RAG with no framework
abstractions in the way.

The point of starting there was to understand what RAG actually *is* before reaching for a
library that hides it. It turns out to be four steps:

1. Embed each chunk of your documents into a vector
2. Embed the user's question the same way
3. Sort chunks by cosine similarity to the question
4. Paste the top matches into the prompt

Everything else the ecosystem sells you is optimisation on top of those four steps.

## Models

| Role | Model |
|------|-------|
| Embedding | [`CompendiumLabs/bge-base-en-v1.5-gguf`](https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf) |
| Language | [`bartowski/Llama-3.2-1B-Instruct-GGUF`](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) |

Dataset: [a list of ~150 cat facts](https://huggingface.co/ngxson/demo_simple_rag_py/blob/main/cat-facts.txt),
one fact per line. Each line is treated as one chunk.

## Run it with Docker (recommended)

Needs Docker and nothing else — no Python, no Ollama installed on the host. The runtime
ships with the app.

```bash
git clone https://github.com/Kasimis/KasRAG.git
cd KasRAG
docker compose up --build
```

Then pull the models once, from a second terminal:

```bash
docker compose exec ollama ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
docker compose exec ollama ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Open <http://localhost:8501>.

The models are about 1.5 GB and live in a named volume, so they are downloaded once and
survive restarts. Give Docker at least 6 GB of memory — the language model is killed with
less.

Two services run side by side: `app` (Streamlit) and `ollama` (the model server). The app
finds the server through the `OLLAMA_HOST` environment variable rather than a hardcoded
address, so the same image runs unchanged wherever the server happens to live.

## Run it without Docker

**1. Install Ollama** from [ollama.com](https://ollama.com) and pull the models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Make sure the Ollama server is running — on Windows it lives in the system tray; otherwise
`ollama serve`.

**2. Clone and install:**

```bash
git clone https://github.com/Kasimis/KasRAG.git
cd KasRAG
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Get the dataset** (if `data/cat-facts.txt` isn't there):

```bash
curl -L -o data/cat-facts.txt https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt
```

**4. Run it:**

```bash
streamlit run main.py
```

The first question triggers embedding of all 150 chunks, which takes a minute or two on CPU.
Every question after that is instant.

## Project structure

```
KasRAG/
├── main.py             # the whole thing: indexing, retrieval, generation, UI
├── data/
│   └── cat-facts.txt   # the dataset, one fact per line
├── Dockerfile          # image for the Streamlit app
├── docker-compose.yml  # app + Ollama, networking, model volume
├── .dockerignore
└── requirements.txt
```

## How it works

The "vector database" is a Python list of `(chunk, embedding)` tuples, and retrieval is a
linear scan computing cosine similarity against every one of them. At 150 chunks that is
instant. At 100,000 it would not be — and that is precisely why FAISS, Chroma and pgvector
exist: approximate nearest-neighbour indexes so you don't compare against every vector.

The index is built inside a function wrapped in `@st.cache_resource`. Streamlit re-executes
the entire script on every interaction, so without that cache the whole dataset would be
re-embedded for every single question.

Similarly, chunking here is trivial because each cat fact is already a self-contained
sentence. Real documents are where chunking becomes the hard part, and where most RAG
quality actually lives.

## Roadmap

- [x] **Streamlit frontend** — replaces the terminal prompt, showing the retrieved chunks and
  their similarity scores alongside each answer.
- [x] **Containerised** — `docker compose up` brings up the app and the model server together,
  with the models persisted in a volume.
- [ ] **Persist the embedding index to disk** — `@st.cache_resource` keeps the vectors for the
  life of the process, but a restart still costs a full re-embed. Pickling them would make
  startup instant.
- [ ] **PDF ingestion** — replace the line-per-fact text file with real documents. This means
  adding a text extraction step and a proper chunking strategy (sliding window with overlap,
  probably), plus keeping page numbers as metadata so answers can cite their source.
- [ ] **Cloud deployment** — Streamlit Community Cloud can't host this: there is no way to run
  the Ollama server or pull GGUF models there, so a deployed app would have nothing to talk
  to. The realistic path is Azure Container Apps with the app and Ollama as sidecar
  containers, and the models baked into a custom Ollama image rather than pulled at runtime,
  since a container filesystem is ephemeral. The alternative — abstracting the model calls so
  a hosted provider can be swapped in — deploys far more easily but gives up the no-API-key
  property that makes this project worth having.
- [ ] Swap the linear scan for a real ANN index once the dataset outgrows it

## Notes

Everything runs locally — no data leaves the machine, and nothing here needs a secret. That
is the point rather than a limitation: local inference keeps confidential material where it
belongs, and the price is paid in RAM, CPU and slower answers instead of per-token billing.

If a hosted provider is added later, its key belongs in a `.env` file or
`.streamlit/secrets.toml`, both already covered by `.gitignore`.

## Credits

- [ngxson](https://huggingface.co/ngxson) for the original guide and the cat-facts dataset
- [Ollama](https://ollama.com) for making local model serving painless
