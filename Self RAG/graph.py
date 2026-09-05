from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    isuse: Literal["useful", "not_useful"]
    use_reason: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    

graph = StateGraph(State)

graph.add_node("decide_retrieval", decide_retrieval)
graph.add_node("generate_direct", generate_direct)
graph.add_node("retrieve", retrieve)

graph.add_node("is_relevant", is_relevant)
graph.add_node("generate_from_context", generate_from_context)
graph.add_node("no_answer_found", no_answer_found)

graph.add_node("is_sup", is_sup)
graph.add_node("revise_answer", revise_answer)


graph.add_node("is_use", is_use)

graph.add_node("rewrite_question", rewrite_question)

graph.add_edge(START, "decide_retrieval")

graph.add_conditional_edges(
    "decide_retrieval",
    route_after_decide,
    {"generate_direct": "generate_direct", "retrieve": "retrieve"},
)

graph.add_edge("generate_direct", END)

graph.add_edge("retrieve", "is_relevant")

graph.add_conditional_edges(
    "is_relevant",
    route_after_relevance,
    {
        "generate_from_context": "generate_from_context",
        "no_answer_found": "no_answer_found"
    },
)

graph.add_edge("no_answer_found", END)

graph.add_edge("generate_from_context", "is_sup")

graph.add_conditional_edges(
    "is_sup",
    route_after_issup,
    {
        "accept_answer": "is_use",      
        "revise_answer": "revise_answer"
    },
)

graph.add_edge("revise_answer", "is_sup")  

graph.add_conditional_edges(
    "is_use",
    route_after_isuse,
    {
        "END": END,
        "rewrite_question": "rewrite_question",
        "no_answer_found": "no_answer_found",
    },
)

graph.add_edge("rewrite_question", "retrieve")

workflow = graph.compile()
workflow
