import sqlite3
import uuid
from typing import Annotated, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


st.set_page_config(
    page_title="RAG & Tools Assistant", page_icon="🤖", layout="wide"
)
st.title("🤖 LangGraph RAG Assistant")

load_dotenv()

DB_NAME = "my_rag.db"

@st.cache_resource
def get_checkpointer():
    con = sqlite3.connect(DB_NAME, check_same_thread=False)
    return SqliteSaver(conn=con)


checkpointer = get_checkpointer()


def get_all_threads():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY rowid DESC")
        threads = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        return threads
    except Exception:
        return []


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

@st.cache_resource
def load_rag_components():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"local_files_only": True},
    )
    loaded_vector_store = FAISS.load_local(
        "faiss_index_dir", embedding, allow_dangerous_deserialization=True
    )
    return loaded_vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )


retriever = load_rag_components()


@tool
def rag_document_fetch(query: str):
    """Retrieve relevant information from the documents
    use this tool when user wants to ask question related to software requirement engineering"""
    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    return {"query": query, "context": context, "metadata": metadata}


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


wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddg_search = DuckDuckGoSearchResults(api_wrapper=wrapper)


@tool
def web_search(query: str) -> str:
    """Search the web for real-time information or news using DuckDuckGo."""
    return ddg_search.run(query)


tools = [rag_document_fetch, calculator, web_search]

@st.cache_resource
def build_graph():
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    model_with_tools = model.bind_tools(tools=tools)

    class chat_state(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]

    def chat_node(state: chat_state) -> chat_state:
        messages = state["messages"]
        try:
            response = model_with_tools.invoke(messages)
        except Exception:
            response = model.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools=tools)

    graph = StateGraph(chat_state)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition, ["tools", END])
    graph.add_edge("tools", "chat")

    return graph.compile(checkpointer=checkpointer)


chat_bot = build_graph()

with st.sidebar:
    st.header(" Chat Sessions")

    if st.button(" New Chat Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.subheader("Previous Threads")

    saved_threads = get_all_threads()

    if not saved_threads:
        st.caption("No past sessions found.")
    else:
        for tid in saved_threads:

            label = f" {tid[:8]}..." if tid == st.session_state.thread_id else f" {tid[:8]}..."
            if st.button(label, key=tid, use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()


config = {"configurable": {"thread_id": st.session_state.thread_id}}

checkpoint_state = chat_bot.get_state(config)
stored_messages = checkpoint_state.values.get("messages", [])

for msg in stored_messages:
    if isinstance(msg, SystemMessage):
        continue
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


system_prompt = SystemMessage(
    content=(
        "You are a helpful assistant with access to document retrieval, a calculator, and web search tools.\n\n"
        "1. For questions about software engineering or requirements, ALWAYS check the 'rag_document_fetch' tool first.\n"
        "2. If the answer IS found in the document, base your response on that retrieved context.\n"
        "3. If the answer IS NOT in the document or the user asks general questions, "
        "use your own internal knowledge or web search to answer clearly, but briefly mention that the information came from general knowledge rather than the provided document."
    )
)

user_query = st.chat_input(
    "Ask a question from your book, calculate something, or search the web..."
)

if user_query:
  
    with st.chat_message("user"):
        st.markdown(user_query)

   
    with st.chat_message("assistant"):
        with st.spinner("Thinking & checking tools..."):
            input_payload = {
                "messages": [system_prompt, HumanMessage(content=user_query)]
            }

            output = chat_bot.invoke(input_payload, config=config)
            final_response = output["messages"][-1].content

            st.markdown(final_response)
            st.rerun()