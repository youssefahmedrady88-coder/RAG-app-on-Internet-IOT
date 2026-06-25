import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from workflow import Workflow

workflow = Workflow()
chat_history = []

def chat(user_message, history):
    global chat_history
    result = workflow.run({
        "query": user_message,
        "chat_history": chat_history,
        "rewritten_query": "",
        "context": [],
        "response": ""
    })
    response = result.get("response", "Sorry, I could not generate a response.")
    context_docs = result.get("context", [])
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response))
    if context_docs:
        sources = "\n\n---\n Sources:\n"
        for doc in context_docs:
            source_name = doc.metadata.get("source", "Unknown")
            sources += f"- {source_name}: {doc.page_content[:200]}...\n"
        response += sources
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Internet and IoT History RAG Chatbot",
    description="Powered by qwen2.5:3b, ChromaDB, LangGraph",
    examples=["What is ARPANET?", "When was the World Wide Web invented?", "What is IoT?"]
)

if __name__ == "__main__":
    demo.launch()
