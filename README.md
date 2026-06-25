# 🌐 Internet & IoT History — RAG Pipeline

A fully local **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the History of the Internet and Internet of Things (IoT), built as part of a practical NLP/LLM exam.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Rewriter     │  ← Rewrites query into a standalone question
│  (qwen2.5:3b)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Dense Retriever    │  ← Finds top-6 relevant chunks from ChromaDB
│  (nomic-embed-text) │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Response Agent     │  ← Generates grounded answer from context
│  (qwen2.5:3b)       │
└─────────┬───────────┘
          │
          ▼
     Final Answer
```

**Stack:** LangGraph · LangChain · ChromaDB · Ollama · Gradio

---

## 📁 Project Structure

```
├── app.py                     # Gradio chat UI
├── workflow.py                # LangGraph pipeline graph
├── agents.py                  # Rewriter, Retriever, Response agents
├── models.py                  # LangGraph State definition
├── prompts.py                 # System prompts + prompt builders
├── Getting_Data_From_Wiki.py  # Fetches Wikipedia data → builds ChromaDB
├── rag_pipeline.py            # Custom RAG pipeline (from-scratch implementation)
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excludes chroma_db/ and other artifacts
└── README.md                  # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone the repository
```bash
git clone https://https://github.com/youssefahmedrady88-coder/RAG-app-on-Internet-IOT
cd RAG-app-on-Internet-IOT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull Ollama models
```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 4. Build the vector database *(run once)*
```bash
python Getting_Data_From_Wiki.py
```
This fetches Wikipedia articles on Internet History & IoT, embeds them using `nomic-embed-text`, and saves the ChromaDB index to `./chroma_db/`.

### 5. Launch the app
```bash
python app.py
```
Then open http://127.0.0.1:7860 in your browser.

## 📚 Dataset

Wikipedia articles fetched automatically via `wikipediaapi`:

| Article | Topic |
|---|---|
| History of the Internet | Origins, ARPANET, milestones |
| Internet of Things | IoT concepts, smart devices |
| ARPANET | First packet-switched network |
| World Wide Web | Tim Berners-Lee, HTTP, HTML |
| Internet Protocol Suite | TCP/IP stack |
| Smart Device | Connected device ecosystem |
| IPv6 | Next-generation addressing |

---

## 🧪 RAG Pipeline (from scratch)

`rag_pipeline.py` implements core RAG concepts manually without high-level abstractions:

- **Text Chunking** — splits documents by word count with configurable overlap
- **Cosine Similarity** — manual vector similarity for retrieval ranking
- **Prompt Engineering** — system prompt constraints to reduce hallucination
- **Decoding Parameters** — experiments with temperature, top-p, top-k

---

## 🔬 Hyperparameter Experiments

| Parameter | Values Tested | Observed Effect |
|---|---|---|
| Temperature | 0.0, 0.2, 0.9 | Higher = more creative but less grounded |
| Top-P | 0.7, 0.9, 1.0 | Controls diversity of token sampling |
| Chunk Size | 200, 500, 1000 words | Smaller = precise, Larger = more context |
| Chunk Overlap | 0, 50, 100 words | Overlap preserves boundary context |

---

## 📝 Notes on `chroma_db/`

The `chroma_db/` directory is **excluded from this repository** (see `.gitignore`) because it is auto-generated. To rebuild it locally, run `Getting_Data_From_Wiki.py` — it fetches the same Wikipedia articles and produces an identical index in under 2 minutes.

---

## 👤 Author

**[Youssef Ahmed]**  
RAG app on The History of internet and IOT  
[Wyise Diploma]
