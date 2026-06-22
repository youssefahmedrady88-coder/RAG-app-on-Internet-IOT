
import wikipediaapi
# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


# ── LangChain integrations ────────────────────────────────────────────────────
from langchain_chroma import Chroma


# ── ChromaDB persistence directory ───────────────────────────────────────────
CHROMA_DIR = "./chroma_db"

# Wikipedia articles that form my dataset
TOPICS = [
    "History of the Internet",
    "Internet of things",
    "ARPANET",
    "World Wide Web",
    "Internet protocol suite",
    "Smart device",
    "IPv6",
]


def fetch_wikipedia_data(topics: list[str]) -> list[Document]:
    """
    Download full article text from Wikipedia and wrap in LangChain Documents.

    LangChain's Document class holds:
        .page_content : str   — the raw text
        .metadata     : dict  — source label, used for citations

    Returns:
        List of LangChain Document objects, one per article.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="WiyseRAGPipeline/1.0 (student assignment)"
    )

    documents = []
    for topic in topics:
        page = wiki.page(topic)
        if not page.exists():
            print(f"[WARNING] Page not found: '{topic}' — skipping.")
            continue

        word_count = len(page.text.split())
        print(f"[INFO] Fetched '{page.title}' ({word_count:,} words)")

        documents.append(Document(
            page_content=page.text,
            metadata={"source": page.title, "url": page.fullurl}
        ))

    return documents

# Chunking Data


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    WHY CHUNKING?
        Embedding models and LLMs have fixed context-window limits.
        A full Wikipedia article (5,000–15,000 words) cannot be embedded
        or fed to a model in one shot. Chunking divides it into segments
        that each fit within these limits while keeping related content
        together.

    Args:
        documents:     List of LangChain Documents (full articles).


    Returns:
        List of LangChain Document as 1 chunk, metadata preserved.
    """
    chunks = documents
    print(
        f"[INFO] Using {len(chunks)} documents as-is (one chunk per article)")
    return chunks


def build_vdb(chunks: list[Document]) -> Chroma:
    """
    Embed all chunks and store them in a persistent ChromaDB vector store.

    WHY ChromaDB INSTEAD OF A PLAIN LIST?
        Plain list → O(N) linear scan every query (slow at scale).
        ChromaDB   → approximate nearest-neighbour index (fast at scale),
                     persists to disk so you don't re-embed on every run,
                     and supports metadata filtering for citation retrieval.

    WHY OllamaEmbeddings?
        Keeps the entire pipeline fully local — no OpenAI key, no cost,
        no data leaving your machine. The same Ollama server that runs
        Gemma 4B also serves the embedding model (nomic-embed-text).

    Args:
        chunks: List of chunked LangChain Documents.

    Returns:
        A Chroma vector store ready for similarity_search().
    """
    print("[INFO] Loading embedding model via Ollama …")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("[INFO] Embedding chunks and building ChromaDB index …")
    vdb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="internet_iot_rag",
    )
    print(f"[INFO] ChromaDB index built and saved to '{CHROMA_DIR}'\n")
    return vdb


def main():
    print("=" * 70)
    print("  WIYSE SCHOOL — LangChain RAG: History of the Internet & IoT")
    print("=" * 70 + "\n")

    # ── Stage 1: Fetch Wikipedia articles ────────────────────────────────────
    print("[STAGE 1/4] Fetching Wikipedia articles …\n")
    documents = fetch_wikipedia_data(TOPICS)
    if not documents:
        print("[ERROR] No documents fetched. Check your internet connection.")
        return

    # ── Stage 2: Chunk documents ─────────────────────────────────────────────
    print("\n[STAGE 2/4] Using each document as one chunk...\n")
    chunks = chunk_documents(documents)

    # ── Stage 3: Build ChromaDB vector index ─────────────────────────────────
    print("\n[STAGE 3/4] Building ChromaDB vector index …\n")
    vdb = build_vdb(chunks)


if __name__ == "__main__":
    main()
