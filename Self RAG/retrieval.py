from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


loader = PyPDFLoader(file_path="Software.pdf") # provide the path of your fiel 
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600, chunk_overlap=150
    )

chunks = splitter.split_documents(docs)


embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

