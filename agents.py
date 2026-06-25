from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from models import State
from prompts import *

llm = OllamaLLM(model='qwen2.5:3b', temperature=0.0)
embedding = OllamaEmbeddings(model='nomic-embed-text')
CHROMA_DIR = './chroma_db'


def rewritten_query_agent(state: State) -> dict:
    user_input = state.get('query', '').strip()
    if len(user_input) < 3:
        return {'rewritten_query': user_input}
    chat_history = state.get('chat_history')
    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(content=query_rewrite_extend(user_input, chat_history))
    ]
    return {'rewritten_query': llm.invoke(messages)}


def retriever_agent(state: State) -> dict:
    rewritten_query = state.get('rewritten_query')
    vdb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)
    dens_ret = vdb.as_retriever(search_kwargs={'k': 2})
    docs = dens_ret.invoke(rewritten_query)
    return {'context': docs}


def response_agent(state: State) -> dict:
    user_input = state.get('rewritten_query')
    context = state.get('context') or []
    context_str = '\n\n---\n\n'.join(
        doc.page_content[:3000] if hasattr(doc, 'page_content') else str(doc)
        for doc in context
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f'User Query: {user_input}\n\nContext:\n{context_str}\n\nAnswer based only on the context above.')
    ]
    return {'response': llm.invoke(messages)}
