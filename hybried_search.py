import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from langchain_community.retrievers import EnsembleRetriever

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env")



pdf_path = "data/my_document.pdf"

loader = PyPDFLoader(pdf_path)

documents = loader.load()

print(f"Loaded documents: {len(documents)}")



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hybrid_search_demo",
    persist_directory="./chroma_db"
)

print("Chroma vector database created.")


vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

bm25_retriever = BM25Retriever.from_documents(
    chunks
)

bm25_retriever.k = 5

print("BM25 retriever created.")


hybrid_retriever = EnsembleRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    weights=[
        0.7,   
        0.3  
    ]
)

query = "What is regression testing?"

retrieved_documents = hybrid_retriever.invoke(query)

print("\n================ RETRIEVED DOCUMENTS ================\n")

for i, doc in enumerate(retrieved_documents, start=1):

    print(f"DOCUMENT {i}")
    print("-" * 50)

    print(doc.page_content[:1000])

    print("\nMetadata:")
    print(doc.metadata)

    print("\n")



llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


prompt = ChatPromptTemplate.from_template(
    """
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you do not have enough information.

Do not make up information.

Context:

{context}

Question:

{question}

Answer:
"""
)


def format_documents(documents):

    formatted_documents = []

    for i, doc in enumerate(documents, start=1):

        formatted_documents.append(
            f"""
Document {i}

Content:
{doc.page_content}

Metadata:
{doc.metadata}
"""
        )

    return "\n\n".join(formatted_documents)



def hybrid_rag(question):

    documents = hybrid_retriever.invoke(question)


    context = format_documents(documents)

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

   

    response = llm.invoke(messages)

    return response.content, documents


while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    answer, documents = hybrid_rag(question)

    print("\n================ ANSWER ================\n")

    print(answer)

    print("\n================ SOURCES ================\n")

    for i, doc in enumerate(documents, start=1):

        print(f"Source {i}")

        print(
            "Page:",
            doc.metadata.get("page", "Unknown")
        )

        print(
            "Content:",
            doc.page_content[:300]
        )

        print("-" * 50)