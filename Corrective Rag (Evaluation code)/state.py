from langchain_core.documents import Document
from typing import TypedDict

class State(TypedDict):
    question : str 
    docs : list[Document]
    
    good_docs : list[Document]
    verdict : str
    reason  : str
    
    
    strips : list[str]
    kept_strips : list[str]
    refined_context : str
    
    answer : str
    
    