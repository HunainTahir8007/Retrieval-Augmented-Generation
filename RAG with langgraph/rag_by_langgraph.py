from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langgraph.prebuilt import ToolNode , tools_condition
from langgraph.graph import StateGraph , START , END
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from typing import Annotated , TypedDict 
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_huggingface import HuggingFaceEmbeddings
import sqlite3
from dotenv import load_dotenv 
from PIL import Image 
import io 
import uuid
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper 
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_core.messages import AIMessage , HumanMessage , SystemMessage
import sys



sys.stdout.reconfigure(errors="ignore") 

con = sqlite3.connect("my_rag.db" , check_same_thread=False)
cheakpoint = SqliteSaver(conn=con)

load_dotenv()
model = ChatGroq(model= "llama-3.3-70b-versatile", temperature=0.2)

loader = PyPDFLoader(file_path="Software.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size = 2000 , chunk_overlap = 200)
chunk = splitter.split_documents(documents=docs)

embeeding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True}
)

vector_store = FAISS.from_documents(chunk , embedding=embeeding)
vector_store.save_local("faiss_index_dir")

loaded_vector_store = FAISS.load_local(
    "faiss_index_dir", 
    embeeding, 
    allow_dangerous_deserialization=True
)
retriever = loaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

@tool 
def rag_document_fetch(query : str):
   """Retrieve relivent infomation form the documents 
   use this tool when user wants to ask question in related to software requirement 
   engineeering"""
   result = retriever.invoke(query)
   context = [doc.page_content for doc in result]
   metadata = [doc.metadata for doc in result]
   return {
       "query": query,
       "context" : context,
       "metadata" : metadata
   }


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations: add, subt, mul, div."""
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "subt":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": "Unsupported operation"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


wrapper = DuckDuckGoSearchAPIWrapper( max_results=5 ) 

ddg_search = DuckDuckGoSearchResults(api_wrapper=wrapper)

@tool
def web_search(query: str) -> str:
    """Search the web for real-time information or news using DuckDuckGo."""
    return ddg_search.run(query)

tools = [rag_document_fetch ,calculator , web_search]
model_with_tools = model.bind_tools(tools=tools)

class chat_state(TypedDict):
    messages : Annotated[list[BaseMessage] , add_messages]


def chat_node(state: chat_state) -> chat_state:
    messages = state["messages"]

    try:
        response = model_with_tools.invoke(messages)
    except Exception as e:
        print(
            f"\n[Tool calling error detected ({type(e).__name__}). Falling back to model knowledge...]\n"
        )
        response = model.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools=tools)


graph = StateGraph(chat_state)

graph.add_node("chat" , chat_node)
graph.add_node("tools" , tool_node)

graph.add_edge(START , "chat")
graph.add_conditional_edges("chat" , tools_condition, ["tools", END])
graph.add_edge("tools" , "chat")

chat_bot =  graph.compile(checkpointer=cheakpoint)

#------------------------------
# png = chat_bot.get_graph().draw_mermaid_png()
# img = Image.open(io.BytesIO(png))
# img.show()

#----------------------------
out = chat_bot.invoke(
    {
        "messages": [
            SystemMessage(
                content=(
                    "You are a strict RAG assistant. "
                    "When answering questions about software engineering or requirements, "
                    "you MUST ALWAYS use the 'rag_document_fetch' tool first. "
                    "Base your answer ONLY on the context returned from the document tool. "
                    "If the information is not in the fetched context, explicitly say: 'This information is not present in the provided document.'"
                )
            ),
            HumanMessage(
                content="Tell me about key components that are included in a software requirement specification document according to the book."
            ),
        ]
    },
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)

print(out['messages'][-1].content)