from langchain_core.documents import Document
from pydantic import BaseModel 
from langchain_core.prompts import ChatPromptTemplate




retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def retrieve(state : State)->State:
    query =state['question']
    retrieved_docs = retriever.invoke(query)
    return {'docs' : retrieved_docs}

class evalutor_model(BaseModel):
    score : float
    reason : str
    
doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
       (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only."
        ),
      ("human","Question: {question}\n\nChunk:\n{chunk}")
    ]
)

upper_threshold = 0.7
lower_threshold = 0.3

eval_chain = doc_eval_prompt | strip_model.with_structured_output(evalutor_model)


def eval_each_doc_node(state :State)-> State:
    q = state['question']
    
    scores : list[float] = []
    reasons : list[str] = []
    good : list[Document] = []
    
    for d in state['docs']:
        out = eval_chain.invoke({"question" : q , "chunk" : d.page_content})
        scores.append(out.score)
        reasons.append(out.reason)
        if out.score > lower_threshold:
            good.append(d)
    
    if any(s > upper_threshold for s in scores):
        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": f"At least one retrieved chunk scored > {upper_threshold}.",
        }

    if len(scores) > 0 and all(s < lower_threshold for s in scores):
        why = "No chunk was sufficient."
        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": f"All retrieved chunks scored < {lower_threshold}. {why}",
        }

    
    why = "Mixed relevance signals."
    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": f"No chunk scored > {upper_threshold}, but not all were < {lower_threshold}. {why}",
    }
    