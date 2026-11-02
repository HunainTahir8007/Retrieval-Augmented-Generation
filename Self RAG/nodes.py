from typing import List, TypedDict, Literal
from langchain_core.documents import Document
from langchain_groq import ChatGroq


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
    
llm = ChatGroq(model="openai/gpt-oss-120b")


def decide_retrieval(state: State):
    decision  = should_retrieve_chain.invoke(
        {"question" : state["question"]}
         )
    
    return {"need_retrieval": decision.should_retrieve}


def generate_direct(state: State):
    out = llm.invoke({"question": state["question"]})
    return {"answer": out.content}


def retrieve(state: State):
    q = state.get("retrieval_query") or state["question"]
    return {"docs": retriever.invoke(q)}

def is_relevant(state: State):
    relevant_docs: List[Document] = []
    for doc in state.get("docs", []):
        decision = relivence_chain.invoke({"question" : state["question"]  , "document" : doc})
        if decision.is_relevant:
            relevant_docs.append(doc)
    return {"relevant_docs": relevant_docs}

def generate_from_context(state: State):
    context = "\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
    res = llm.invoke({"question" : state["question"] , "context" : context})
    return {"answer": res.content, "context": context}

def is_sup(state: State):
    decision = issupported_chain.invoke({
            "question" :state["question"],
            "answer" :state['answer'],
            "context" : state['context']
            }
        )
    
    return {"issup": decision.issup, "evidence": decision.evidence}

def revise_answer(state: State):
    out = revise_llm.invoke({
    
            "question":state["question"],
            "answer":state.get("answer", ""),
            "context" : state.get("context", "")
    }
        )
    
    return {
        "answer": out.content,
        "retries": state.get("retries", 0) + 1,  
    }

def is_use(state: State):
    decision = isuse_chain.invoke({
        
           "question":state["question"],
            "answer":state.get("answer", "")
    }   
    )
    return {"isuse": decision.isuse, "use_reason": decision.reason}

def rewrite_question(state: State):
    decision = rewrite_chain.invoke({       
            "question" :state["question"],
            "retrieval_query" :state.get("retrieval_query", ""),
            "answer" : state.get("answer", "")
    }
        )
    return {
        "retrieval_query": decision.retrieval_query,
        "rewrite_tries": state.get("rewrite_tries", 0) + 1,
        "docs": [],
        "relevant_docs": [],
        "context": "",
    }
    

