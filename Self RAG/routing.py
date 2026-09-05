

def route_after_decide(state: State) :
    return "retrieve" if state["need_retrieval"] else "generate_direct"

def route_after_relevance(state: State):
    if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
        return "generate_from_context"
    return "no_answer_found"

def route_after_issup(state: State) -> Literal["accept_answer", "revise_answer"]:
    if state['issup'] == "fully_supported":
        return "accept_answer"

    if state['retries'] >= MAX_RETRIES:
        return "accept_answer"  

    return "revise_answer"


MAX_REWRITE_TRIES = 3 

def route_after_isuse(state: State) -> Literal["END", "rewrite_question", "no_answer_found"]:
    if state.get("isuse") == "useful":
        return "END"

    if state.get("rewrite_tries", 0) >= MAX_REWRITE_TRIES:
        return "no_answer_found"

    return "rewrite_question"

