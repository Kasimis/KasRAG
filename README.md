# myrag

A minimal Retrieval-Augmented Generation system in plain Python — no LangChain, no vector
database service, and **no API keys**. Both models run locally through
[Ollama](https://ollama.com), so the whole thing works offline once the models are pulled.

Ask a question, the script finds the most relevant facts from a local text file by cosine
similarity, pastes them into the prompt, and a small Llama model answers from that context
alone.

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

## Setup

**1. Install Ollama** from [ollama.com](https://ollama.com) and pull the models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

Make sure the Ollama server is running — on Windows it lives in the system tray; otherwise
`ollama serve`.

**2. Clone and install:**

```bash
git clone https://github.com/Kasimis/myrag.git
cd myrag
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
python main.py
```

It embeds all 150 chunks on startup, which takes a minute or two on CPU, then prompts you
for a question.

## Project structure

```
myrag/
├── main.py           # the whole thing: indexing, retrieval, generation
├── data/
│   └── cat-facts.txt # the knowledge base, one fact per line
└── requirements.txt
```

## How it works

The "vector database" is a Python list of `(chunk, embedding)` tuples, and retrieval is a
linear scan computing cosine similarity against every one of them. At 150 chunks that is
instant. At 100,000 it would not be — and that is precisely why FAISS, Chroma and pgvector
exist: approximate nearest-neighbour indexes so you don't compare against every vector.

Similarly, chunking here is trivial because each cat fact is already a self-contained
sentence. Real documents are where chunking becomes the hard part, and where most RAG
quality actually lives.

## Roadmap

- [ ] **Streamlit frontend** — replace the terminal prompt with a proper chat UI, showing the
  retrieved chunks and their similarity scores alongside each answer.
- [ ] **Cache the embedding index** — pickle the vectors to disk so startup isn't a two-minute
  wait every single run.
- [ ] **PDF ingestion** — replace the line-per-fact text file with real documents. This means
  adding a text extraction step and a proper chunking strategy (sliding window with overlap,
  probably), plus keeping page numbers as metadata so answers can cite their source.
- [ ] **Cloud deployment** — currently local-only by necessity. Streamlit Community Cloud
  can't host this as-is: there's no way to install the Ollama server or pull GGUF models
  there, so a deployed app would have nothing to talk to. Two viable paths: a VM or container
  host (Fly.io, Railway, a VPS) where Ollama can actually run, or an abstraction over the
  model calls so a hosted provider can be swapped in when Ollama isn't reachable. The second
  is easier to deploy but gives up the no-API-key property that makes this project nice.
- [ ] Swap the linear scan for a real ANN index once the corpus outgrows it

## Notes

Everything runs on `localhost` — no data leaves the machine, and nothing here needs a secret.
If a hosted provider is added later, its key belongs in a `.env` file or
`.streamlit/secrets.toml`, both already covered by `.gitignore`.

## Credits

- [ngxson](https://huggingface.co/ngxson) for the original guide and the cat-facts dataset
- [Ollama](https://ollama.com) for making local model serving painless
