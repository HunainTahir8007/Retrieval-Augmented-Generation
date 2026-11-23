import re
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel 

def decompose_to_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]
class keepordrop(BaseModel):
    keep :bool
filter_prompt = ChatPromptTemplate.from_messages(
    [
         ("system",
            "You are a strict relevance filter."
            "Return keep=true only if the sentence directly helps answer the question."
            "Use ONLY the sentence. Output JSON only."),
         
         ("human", "Question: {question}\n\nSentence:\n{sentence}")
    ]
)

filter_chain = filter_prompt | strip_model.with_structured_output(keepordrop)

def refine(state : State) -> State:
    q = state['question']
    context = "\n\n".join(d.page_content for d in state['good_docs']).strip()
    strips = decompose_to_sentences(context)
    
    kept  : list[str] = []
    for s in strips:
        if filter_chain.invoke({"question" : q , "sentence" : s }).keep:
            kept.append(s)
        
    refined_context ="\n".join(kept).strip()
    return {
        
        "strips":strips , 
        "kept_strips" : kept ,
        "refined_context" : refined_context
        
            }
    
    