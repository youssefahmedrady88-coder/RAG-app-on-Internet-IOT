from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    rewritten_query: str
    query: str
    chat_history: Annotated[list, add_messages]
    context: Optional[list[str]]
    response: str