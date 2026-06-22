from langgraph.graph import StateGraph, START, END
from agents import *
from models import State


class Workflow:
    def __init__(self):
        self.rewrite_query_agent = rewritten_query_agent
        self.retriever_agent = retriever_agent
        self.response_agent = response_agent

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("rewrite_query", self.rewrite_query_agent)
        graph.add_node("retriever_agent", self.retriever_agent)
        graph.add_node("reponse_agent", self.response_agent)

        graph.add_edge(START, "rewrite_query")
        graph.add_edge("rewrite_query", "retriever_agent")
        graph.add_edge("retriever_agent", "reponse_agent")
        graph.add_edge("reponse_agent", END)

        return graph.compile()

    def run(self, initial_state: State):
        graph = self._build_graph()
        result = graph.invoke(initial_state)
        return result
