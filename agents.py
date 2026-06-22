# ── LangChain core ────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage


# ── LangChain integrations ────────────────────────────────────────────────────
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma


from models import State
from prompts import *


# ── Ollama settings ───────────────────────────────────────────────────────────
llm = OllamaLLM(model='qwen2.5:3b', temperature=0.0)
embedding = OllamaEmbeddings(model="nomic-embed-text")

CHROMA_DIR = "./chroma_db"


def rewritten_query_agent(state: State) -> dict:
    user_input = state.get("query")
    chat_hsitory = state.get('chat_history')

    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=query_rewrite_extend(user_input, chat_hsitory))
    ]
    rewritten_query = llm.invoke(messages)

    return {"rewritten_query": rewritten_query}


def response_agent(state: State) -> dict:
    """
    Response Agent that gives the response to the user
    """
    user_input = state.get("rewritten_query")
    chat_history = state.get('chat_history', [])
    context = state.get("context", [])

    # Format docs to readable text before passing to prompt
    context_str = "\n\n".join(
        [doc.page_content for doc in context]
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=system_prompt_extend(
            user_input, chat_history, context_str))
    ]

    final_response = llm.invoke(messages)

    return {"response": final_response}


def retriever_agent(state: State) -> dict:
    """
    Retriever of the RAG
    """
    rewritten_query = state.get("rewritten_query")

    vdb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )
    dens_ret = vdb.as_retriever(search_kwargs={'k': 6})
    docs = dens_ret.invoke(rewritten_query)

    return {"context": docs}
