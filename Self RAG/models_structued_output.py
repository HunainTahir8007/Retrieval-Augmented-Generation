from typing import List,  Literal
from pydantic import BaseModel, Field


class RetrieveDecision(BaseModel):
    should_retrieve: bool = Field(description="True if external documents are needed to answer reliably, else False."
    )
    
should_retrieve_llm = llm.with_structured_output(RetrieveDecision)
should_retrieve_chain  = decide_retrieval_prompt | should_retrieve_llm

class RelevanceDecision(BaseModel):
    is_relevant: bool = Field(
        description="True ONLY if the document contains info that can directly answer the question."
    )
    
relevance_llm = llm.with_structured_output(RelevanceDecision)
relivence_chain = is_relevant_prompt | relevance_llm

class Issupported(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str] = Field(default_factory=list)

issupported_llm = llm.with_structured_output(Issupported)
issupported_chain = issup_prompt | issupported_llm

class Isuse(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str = Field( description="Short reason in 1 line.")

isuse_llm = llm.with_structured_output(Isuse)
isuse_chain = isuse_prompt | isuse_llm 

class Rewrite(BaseModel):
    retrieval_query: str = Field(
        description="Rewritten query optimized for vector retrieval against internal company PDFs."
    )
    
rewrite_llm = llm.with_structured_output(Rewrite)
rewrite_chain = rewrite_for_retrieval_prompt | rewrite_llm

