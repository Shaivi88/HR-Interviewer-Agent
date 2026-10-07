from langgraph.graph import StateGraph, END

from pipeline.state import PipelineState
from pipeline.nodes import extract_node, score_node, rank_node, question_gen_node


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("extract", extract_node)
    graph.add_node("score", score_node)
    graph.add_node("rank", rank_node)
    graph.add_node("question_gen", question_gen_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "score")
    graph.add_edge("score", "rank")
    graph.add_edge("rank", "question_gen")
    graph.add_edge("question_gen", END)

    return graph.compile()
