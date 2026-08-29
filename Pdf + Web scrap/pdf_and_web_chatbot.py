from langchain_openai import ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader 
import streamlit as st 
from pypdf import PdfReader


OPENAI_API_KEY = ""
parser = StrOutputParser()

def create_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vector_store, len(chunks)
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def answer_question(question , vector):
    llm = ChatOpenAI(api_key=OPENAI_API_KEY , model="gpt-4o-mini" , temperature= 0.4)
    retriver = vector.as_retriever(search_kwargs={"k":4} )
    prompt = PromptTemplate.from_template("""Answer the question using ONLY the context below.
If answer not found, say "This information is not in the provided content."
 
Context:
{context}
 
Question: {question}
 
Answer:""")
    chain = ({
    
     "context" :   retriver | (lambda docs: "\n\n".join([d.page_content for d in docs])) ,
     "question" :  RunnablePassthrough()
    }
             | prompt | llm | parser
        
    )
    return chain.invoke(question)



def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n[Page {page_num + 1}]\n{page_text}"

    return text


def process(pdf_files):
    documents= []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        
        if text.strip():
            doc = Document(page_content=text , metadata = {"source":pdf_file.name , "type": "Pdf"})
            documents.append(doc)
            
    
    return documents

#------------------------------------Web loading -----------------------------------------
def process_url(urls):
    documents = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
              doc.metadata["source"] = url
              doc.metadata["type"] = "website"
            documents.extend(docs)
        except Exception as e:
            st.warning(f"Could not load: {url}")
    return documents


def main():
 
    st.set_page_config(
        page_title="Multi-Source RAG",
        page_icon="📚",
        layout="centered"
    )
 
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1a1a2e;
        }
        .sub-title {
            font-size: 1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        ..answer-box {
    background: #000000;
    color: #ffffff;
    border-left: 4px solid #0066cc;
    padding: 1.2rem;
    border-radius: 0.5rem;
    margin-top: 1rem;
}
    </style>
    """, unsafe_allow_html=True)
 
    st.markdown('<div class="main-title">📚 Multi-Source RAG App</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ask questions from multiple PDFs or websites using AI</div>', unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["📄 PDF RAG", "🌐 Website RAG"])

    with tab1:
 
        st.subheader("📄 Ask Questions from PDFs")
        st.caption("Upload one or more PDF files — the AI will answer questions from all of them together")
 
        pdf_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )
 
        if pdf_files:
            st.markdown("**Uploaded files:**")
            for pdf in pdf_files:
                st.caption(f"📄 {pdf.name}")
 
        if st.button("Process PDFs", type="primary", key="pdf_process"):
            if not pdf_files:
                st.error("Please upload at least one PDF file")
            else:
      
                with st.spinner("Reading PDFs..."):
                    documents = process(pdf_files)
 
                if not documents:
                    st.error("Could not extract text. Make sure PDFs are not scanned images.")
                else:
               
                    with st.spinner("Creating vector store..."):
                        vector_store, chunk_count = create_vector_store(documents)
 
                    st.session_state.pdf_vector_store = vector_store
                    st.session_state.pdf_ready = True
 
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("PDFs Processed", len(documents))
                    with col2:
                        st.metric("Chunks Created", chunk_count)
 
                    st.success("Ready! Ask your questions below")
 

        if st.session_state.get("pdf_ready"):
            st.markdown("---")
            st.subheader("Ask a Question")
 
            question = st.text_input(
                "Your question:",
                placeholder="What is the main topic? What does it say about...?",
                key="pdf_q"
            )
 
            if st.button("Get Answer", key="pdf_ans"):
                if not question:
                    st.warning("Please type a question first")
                else:
                    with st.spinner("Thinking..."):
                        answer = answer_question(
                            question,
                            st.session_state.pdf_vector_store
                        )
 
                    st.markdown("**Answer:**")
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )
 
            if st.button("Clear", key="pdf_clear"):
                st.session_state.pdf_vector_store = None
                st.session_state.pdf_ready = False
                st.rerun()
 
 
    with tab2:
 
        st.subheader("🌐 Ask Questions from Websites")
        st.caption("Enter one or more URLs — the AI will scrape and answer questions from all pages")
 
        # URL input
        urls_input = st.text_area(
            "Enter website URLs (one per line):",
            placeholder="https://en.wikipedia.org/wiki/Machine_learning\nhttps://another-site.com",
            height=150,
            key="url_input"
        )
 
        if urls_input.strip():
            url_count = len([u for u in urls_input.strip().split("\n") if u.strip()])
            st.caption(f"{url_count} URL(s) entered")
 
        if st.button("Process Websites", type="primary", key="web_process"):
            if not urls_input.strip():
                st.error("Please enter at least one URL")
            else:
                urls = [u.strip() for u in urls_input.strip().split("\n") if u.strip()]
 
                with st.spinner(f"Scraping {len(urls)} website(s)..."):
                    documents = process_url(urls)
 
                if not documents:
                    st.error("Could not scrape any websites. Check your URLs.")
                else:
                    st.markdown("**Successfully loaded:**")
                    for doc in documents:
                        st.caption(f"✅ {doc.metadata['source']}")
 
                    with st.spinner("Creating vector store..."):
                        vector_store, chunk_count = create_vector_store(documents)
 
                    st.session_state.web_vector_store = vector_store
                    st.session_state.web_ready = True
 
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pages Loaded", len(documents))
                    with col2:
                        st.metric("Chunks Created", chunk_count)
 
                    st.success("Ready! Ask your questions below")
 
        if st.session_state.get("web_ready"):
            st.markdown("---")
            st.subheader("Ask a Question")
 
            question = st.text_input(
                "Your question:",
                placeholder="What does this website say about...?",
                key="web_q"
            )
 
            if st.button("Get Answer", key="web_ans"):
                if not question:
                    st.warning("Please type a question first")
                else:
                    with st.spinner("Thinking..."):
                        answer = answer_question(
                            question,
                            st.session_state.web_vector_store
                        )
 
                    st.markdown("**Answer:**")
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )
 
            if st.button("Clear", key="web_clear"):
                st.session_state.web_vector_store = None
                st.session_state.web_ready = False
                st.rerun()
 
 
if __name__ == "__main__":
    main()