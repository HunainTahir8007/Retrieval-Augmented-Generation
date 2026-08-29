import streamlit as st 
import torch 
import numpy as np 
from PIL import Image 
from io import BytesIO
import base64 
from transformers import CLIPModel , CLIPProcessor

from langchain_community.document_loaders import (
    PyPDFLoader , Docx2txtLoader , CSVLoader , WebBaseLoader , 
    WikipediaLoader , YoutubeLoader ,  UnstructuredImageLoader ,
    TextLoader , UnstructuredExcelLoader , JSONLoader , UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_core.output_parsers import StrOutputParser
import tempfile 
import os 


GROQ_API_KEY = ""
LLM_Model = "llama-3.3-70b-versatile"
ClIP_MODEL = "openai/clip-vit-base-patch32"
parser = StrOutputParser()


class CLIPEmbeedings(Embeddings):
    def __init__(self ,  model_name = ClIP_MODEL):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
    
    def embed_documents(self, texts : list[str]):
        embeeddings = []
        batch_size =32
        for i in range(0 , len(texts) , batch_size):
            batch = texts[i:1  + batch_size]
            inputs = self.processor(
                text=batch ,
                return_tensors='pt',
                padding = True,
                truncation = True,
                max_length = 77
            ).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim =-1 , keepdim = True)
                
            embeeddings.extend(text_features.cpu().numpy().tolist())
        
        return embeeddings
    
    def embed_query(self, text):
        return self.embed_documents([text][0])
    
    def embed_image(self , pil_image : Image.Image):
        inputs = self.processor(
            images=pil_image ,
            return_tensors ="pt"
        ).to(self.device)
        with torch.no_grad():
        
          image_features = self.model.get_image_features(**inputs)
          image_features = image_features / image_features.norm(dim=-1 , keepdim=True)
        
        return image_features.cpu().numpy().tolist()[0]
    
    def compute_similarity(self , vec1 : list[float], vec2 : list[float]):
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    


class ImageStore:
    def __init__(self, clip_embeddings : CLIPEmbeedings):
        self.clip = clip_embeddings
        self.images = []
        self.image_metadata = []
    
    def add_image(self , pil_image : Image.Image , metadata :dict):
        embedding = self.clip.embed_image(pil_image)
        
        idx = len(self.images)
        self.images.append(pil_image)
        self.image_metadata.append({**metadata, "image_idx": idx})
        return embedding , idx
    
    def search_by_text(self , query_text :str , k : int =4):
        if not self.images:
            return[]
        query_vec = np.array(self.clip.embed_query(query_text))
 
        image_embeddings = [
            np.array(self.clip.embed_image(img))
            for img in self.images
        ]
 
        similarities = [
            float(np.dot(query_vec, img_vec) /
                  (np.linalg.norm(query_vec) * np.linalg.norm(img_vec)))
            for img_vec in image_embeddings
        ]
 
        sorted_indices = np.argsort(similarities)[::-1][:k]
 
        results = []
        for idx in sorted_indices:
            results.append({
                "image": self.images[idx],
                "metadata": self.image_metadata[idx],
                "similarity": similarities[idx]
            })
 
        return results
 
 

@st.cache_resource
def load_clip_model():
    return CLIPEmbeedings(ClIP_MODEL)


def create_text_vector_store(documents, clip_embeddings):
   
    if not documents:
        return None, 0
 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
 
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=clip_embeddings  
    )
 
    return vector_store, len(chunks)

def answer_question_from_text(question, vector_store):
    """RAG chain for text documents — same as before."""
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_Model,
        temperature=0.2
    )
 
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
 
    prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below.
If not found, say "This was not found in the provided content."
 
Context:
{context}
 
Question: {question}
 
Answer:""")
 
    chain = (
        {
            "context": retriever | (lambda docs: "\n\n---\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in docs
            ])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )
 
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))
 
    return answer, sources
 
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name
 


#--------------------------------------------data loaders ---------------------------------

def load_pdfs(pdf_files):
    documents = []
    for pdf_file in pdf_files:
        tmp_path = save_uploaded_file(pdf_file)
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = pdf_file.name
                doc.metadata["type"] = "pdf"
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    return documents
 
 
def load_word_docs(word_files):
    documents = []
    for word_file in word_files:
        tmp_path = save_uploaded_file(word_file)
        try:
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = word_file.name
                doc.metadata["type"] = "word"
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    return documents
 
 
def load_csvs(csv_files):
    documents = []
    for csv_file in csv_files:
        tmp_path = save_uploaded_file(csv_file)
        try:
            loader = CSVLoader(file_path=tmp_path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = csv_file.name
                doc.metadata["type"] = "csv"
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    return documents
 
 
def load_websites(urls):
    documents = []
    failed = []
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
        except Exception:
            failed.append(url)
    return documents, failed
 
 
def load_wikipedia(query, num_docs=2):
    try:
        loader = WikipediaLoader(
            query=query,
            load_max_docs=num_docs,
            doc_content_chars_max=3000
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["type"] = "wikipedia"
            doc.metadata["source"] = f"Wikipedia: {doc.metadata.get('title', query)}"
        return docs
    except Exception:
        return []
 
def load_youtube(video_id):
    try:
        loader = YoutubeLoader.from_youtube_url(
            f"https://www.youtube.com/watch?v={video_id}",
            language=["en"]
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = f"YouTube: {video_id}"
            doc.metadata["type"] = "youtube"
        return docs
    except Exception:
        return []
 
 
 
def show_answer(answer, sources=None):
    st.markdown("**Answer:**")
    st.markdown(
        f"""<div style="background:#f0f7ff;border-left:4px solid
        #0066cc;padding:1rem;border-radius:0.5rem">{answer}</div>""",
        unsafe_allow_html=True
    )
    if sources:
        with st.expander("Sources"):
            for src in sources:
                st.caption(f"• {src}")
 
 
def text_qa_section(session_key, placeholder="Ask a question..."):
    """Reusable Q&A section for text tabs."""
    if st.session_state.get(session_key):
        st.markdown("---")
        q = st.text_input("Your question:", placeholder=placeholder,
                          key=f"{session_key}_q")
        if st.button("Get Answer", key=f"{session_key}_ans"):
            if q:
                with st.spinner("Searching with CLIP..."):
                    ans, srcs = answer_question_from_text(
                        q, st.session_state[session_key]
                    )
                show_answer(ans, srcs)
            else:
                st.warning("Type a question first")
 
 
 
def main():
    st.set_page_config(
        page_title="CLIP Multimodal RAG",
        page_icon="🖼️",
        layout="wide"
    )
 
    st.title("🖼️ CLIP Multimodal RAG")
    st.caption("Uses CLIP to embed text AND images in the same vector space")
 
    with st.expander("🧠 How CLIP is different from all-MiniLM-L6-v2"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Previous app (all-MiniLM-L6-v2):**")
            st.code("""
Text → embedding (384-dim)
Image → ❌ Cannot embed images
 
Text search: ✅
Image search by text: ❌
            """)
        with col2:
            st.markdown("**This app (CLIP):**")
            st.code("""
Text  → embedding (512-dim) ✅
Image → embedding (512-dim) ✅
Both in SAME vector space!
 
Text search: ✅
Image search by text: ✅ ← KEY DIFFERENCE
            """)
 
    if GROQ_API_KEY == "your_groq_api_key_here":
        st.error("Set your GROQ_API_KEY. Get free at console.groq.com")
        st.stop()
 
   
    with st.spinner("Loading CLIP model (first time only, ~300MB)..."):
        clip = load_clip_model()
    st.success("✅ CLIP model loaded")
 
    if "image_store" not in st.session_state:
        st.session_state.image_store = ImageStore(clip)
 
    tabs = st.tabs([
        "🖼️ Image Search (CLIP Magic)",
        "📄 PDF",
        "📝 Word",
        "📊 CSV",
        "🌐 Website",
        "📖 Wikipedia",
        "🎥 YouTube",
        "🔍 Text + Image Search"
    ])
 
 
    with tabs[0]:
        st.subheader("🖼️ Search Images Using Text (CLIP Only)")
        st.caption("""
        This tab shows CLIP's unique power:
        Upload images → search them using text descriptions.
        all-MiniLM-L6-v2 CANNOT do this — only CLIP can.
        """)
 
        image_files = st.file_uploader(
            "Upload images to search through:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="clip_img_upload"
        )
 
        if image_files:
            if st.button("Add Images to Store", type="primary"):
                for img_file in image_files:
                    pil_img = Image.open(img_file)
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")
 
                    st.session_state.image_store.add_image(
                        pil_img,
                        metadata={
                            "source": img_file.name,
                            "type": "image"
                        }
                    )
 
                n_images = len(st.session_state.image_store.images)
                st.success(f"✅ {n_images} images in store")
 
        if st.session_state.image_store.images:
            n_stored = len(st.session_state.image_store.images)
            st.info(f"📦 Image store contains {n_stored} images")
 
            with st.expander(f"View all {n_stored} stored images"):
                cols = st.columns(min(n_stored, 4))
                for i, img in enumerate(st.session_state.image_store.images):
                    with cols[i % 4]:
                        meta = st.session_state.image_store.image_metadata[i]
                        st.image(img, caption=meta.get("source", f"Image {i+1}"),
                                 use_column_width=True)
 
            st.markdown("---")
            st.subheader("🔍 Search by Text Description")
            st.caption("Type what you're looking for — CLIP finds matching images")
 
            text_query = st.text_input(
                "Describe what you want to find:",
                placeholder="a dog running, a red car, a chart with bars, text about AI...",
                key="img_search_q"
            )
 
            k = st.slider("Number of results:", 1, min(5, n_stored), min(3, n_stored))
 
            if st.button("Search Images", type="primary", key="img_search_btn"):
                if not text_query:
                    st.warning("Enter a search description")
                else:
                    with st.spinner("Searching with CLIP..."):
                        results = st.session_state.image_store.search_by_text(
                            text_query, k=k
                        )
 
                    st.markdown(f"**Top {len(results)} matches for: '{text_query}'**")
                    cols = st.columns(len(results))
                    for i, result in enumerate(results):
                        with cols[i]:
                            st.image(
                                result["image"],
                                caption=f"Score: {result['similarity']:.3f}\n{result['metadata'].get('source', '')}",
                                use_column_width=True
                            )
 
                    st.markdown("**Similarity scores:**")
                    for result in results:
                        score = result['similarity']
                        source = result['metadata'].get('source', 'Unknown')
                        st.progress(
                            min(score, 1.0),
                            text=f"{source}: {score:.3f}"
                        )
 
 
 
    with tabs[1]:
        st.subheader("📄 PDF RAG with CLIP")
        st.caption("Same as before but uses CLIP text encoder instead of all-MiniLM-L6-v2")
 
        pdf_files = st.file_uploader("Upload PDFs", type=["pdf"],
                                      accept_multiple_files=True, key="pdf_up")
 
        if st.button("Process PDFs", type="primary", key="pdf_btn"):
            if not pdf_files:
                st.error("Upload at least one PDF")
            else:
                with st.spinner("Loading PDFs..."):
                    docs = load_pdfs(pdf_files)
                if docs:
                    with st.spinner("Embedding with CLIP text encoder..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.pdf_vs = vs
                    st.metric("Pages", len(docs))
                    st.metric("Chunks", n)
                    st.success("Ready!")
 
        text_qa_section("pdf_vs", "What does this PDF say about...?")
 
 
    with tabs[2]:
        st.subheader("📝 Word RAG with CLIP")
 
        word_files = st.file_uploader("Upload Word files", type=["docx"],
                                       accept_multiple_files=True, key="word_up")
 
        if st.button("Process Word", type="primary", key="word_btn"):
            if not word_files:
                st.error("Upload at least one Word file")
            else:
                with st.spinner("Loading Word docs..."):
                    docs = load_word_docs(word_files)
                if docs:
                    with st.spinner("Embedding with CLIP..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.word_vs = vs
                    st.metric("Documents", len(docs))
                    st.metric("Chunks", n)
                    st.success("Ready!")
 
        text_qa_section("word_vs", "What does this document say about...?")
 
 
    with tabs[3]:
        st.subheader("📊 CSV RAG with CLIP")
 
        csv_files = st.file_uploader("Upload CSV files", type=["csv"],
                                      accept_multiple_files=True, key="csv_up")
 
        if st.button("Process CSV", type="primary", key="csv_btn"):
            if not csv_files:
                st.error("Upload at least one CSV")
            else:
                with st.spinner("Loading CSV..."):
                    docs = load_csvs(csv_files)
                if docs:
                    with st.spinner("Embedding with CLIP..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.csv_vs = vs
                    st.metric("Rows", len(docs))
                    st.metric("Chunks", n)
                    st.success("Ready!")
 
        text_qa_section("csv_vs", "Which row has the highest...?")
 
 
    with tabs[4]:
        st.subheader("🌐 Website RAG with CLIP")
 
        urls = st.text_area("Enter URLs (one per line):", height=100, key="web_urls")
 
        if st.button("Process Websites", type="primary", key="web_btn"):
            if not urls.strip():
                st.error("Enter at least one URL")
            else:
                url_list = [u.strip() for u in urls.strip().split("\n") if u.strip()]
                with st.spinner("Scraping websites..."):
                    docs, failed = load_websites(url_list)
                for f in failed:
                    st.warning(f"Failed: {f}")
                if docs:
                    with st.spinner("Embedding with CLIP..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.web_vs = vs
                    st.metric("Pages", len(docs))
                    st.metric("Chunks", n)
                    st.success("Ready!")
 
        text_qa_section("web_vs", "What does this website say about...?")
 
 
    with tabs[5]:
        st.subheader("📖 Wikipedia RAG with CLIP")
 
        query = st.text_input("Search Wikipedia:", key="wiki_q")
        n_articles = st.slider("Articles:", 1, 5, 2)
 
        if st.button("Search & Load", type="primary", key="wiki_btn"):
            if not query:
                st.error("Enter a search query")
            else:
                with st.spinner("Loading Wikipedia..."):
                    docs = load_wikipedia(query, n_articles)
                if docs:
                    with st.spinner("Embedding with CLIP..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.wiki_vs = vs
                    for doc in docs:
                        st.caption(f"✅ {doc.metadata.get('source')}")
                    st.metric("Articles", len(docs))
                    st.metric("Chunks", n)
                    st.success("Ready!")
                else:
                    st.error("No articles found")
 
        text_qa_section("wiki_vs", "What does Wikipedia say about...?")
 
 
    with tabs[6]:
        st.subheader("🎥 YouTube RAG with CLIP")
 
        video_id = st.text_input("YouTube Video ID:", key="yt_id")
 
        if st.button("Load Video", type="primary", key="yt_btn"):
            if not video_id:
                st.error("Enter a Video ID")
            else:
                with st.spinner("Loading transcript..."):
                    docs = load_youtube(video_id)
                if docs:
                    with st.spinner("Embedding with CLIP..."):
                        vs, n = create_text_vector_store(docs, clip)
                    st.session_state.yt_vs = vs
                    st.metric("Words", len(docs[0].page_content.split()))
                    st.metric("Chunks", n)
                    st.success("Ready!")
                else:
                    st.error("Could not load transcript")
 
        text_qa_section("yt_vs", "What did the video say about...?")
 
    with tabs[7]:
        st.subheader("🔍 Combined Text + Image Search")
        st.caption("""
        Search across BOTH text documents AND images simultaneously.
        This is true multimodal RAG — one query searches everything.
        """)
 
        if not st.session_state.image_store.images:
            st.warning("No images loaded yet. Go to Tab 1 and upload some images first.")
 
        text_sources = {
            "PDF": st.session_state.get("pdf_vs"),
            "Word": st.session_state.get("word_vs"),
            "CSV": st.session_state.get("csv_vs"),
            "Website": st.session_state.get("web_vs"),
            "Wikipedia": st.session_state.get("wiki_vs"),
            "YouTube": st.session_state.get("yt_vs"),
        }
 
        available_text = {k: v for k, v in text_sources.items() if v is not None}
 
        if available_text:
            st.info(f"Text sources available: {', '.join(available_text.keys())}")
        else:
            st.warning("No text sources loaded. Process some content in other tabs first.")
 
        combined_query = st.text_input(
            "Search everything:",
            placeholder="machine learning concepts, charts, diagrams...",
            key="combined_q"
        )
 
        if st.button("Search All Sources", type="primary", key="combined_btn"):
            if not combined_query:
                st.warning("Enter a search query")
            else:
                col1, col2 = st.columns(2)
 
                # Search images
                with col1:
                    st.markdown("**🖼️ Matching Images:**")
                    if st.session_state.image_store.images:
                        with st.spinner("Searching images..."):
                            img_results = st.session_state.image_store.search_by_text(
                                combined_query, k=3
                            )
                        for result in img_results:
                            st.image(
                                result["image"],
                                caption=f"Score: {result['similarity']:.3f} | {result['metadata'].get('source', '')}",
                                use_column_width=True
                            )
                    else:
                        st.info("No images in store")
 
                with col2:
                    st.markdown("**📝 Matching Text:**")
                    if available_text:
                        first_vs = list(available_text.values())[0]
                        with st.spinner("Searching text..."):
                            ans, srcs = answer_question_from_text(
                                combined_query, first_vs
                            )
                        st.markdown(
                            f"""<div style="background:#f0f7ff;border-left:4px solid
                            #0066cc;padding:1rem;border-radius:0.5rem">{ans}</div>""",
                            unsafe_allow_html=True
                        )
                        with st.expander("Sources"):
                            for src in srcs:
                                st.caption(f"• {src}")
                    else:
                        st.info("No text sources loaded")
 
 
if __name__ == "__main__":
    main()