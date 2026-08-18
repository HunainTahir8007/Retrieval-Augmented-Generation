
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import re

from youtube_transcript_api import YouTubeTranscriptApi  # for transcript fetching of video

load_dotenv()

GROQ_API_KEY = ""
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
)
Embeed_model = "sentence-transformers/all-MiniLM-L6-v2"
Chat_LLM = "llama-3.3-70b-versatile"
Chunk_size = 1000
Chunk_overlap = 200
TOP_K_Results = 4


def get_video_transcript(video_id: str):
    """Fetch YouTube captions using youtube_transcript_api."""
    try:
        yyt_api = YouTubeTranscriptApi()
        transcript_list = yyt_api.fetch(
            video_id=video_id,
            languages=["en", "en-US", "en-GB"]
        )
        full_transcript = " ".join([result.text for result in transcript_list])
        return full_transcript
    except Exception:
        return None


def load_trascript(video_id: str):
    """
    Load transcript only from YouTube captions.
    Returns (transcript, source_method) or (None, None) if not available.
    """
    transcript = get_video_transcript(video_id)
    if transcript:
        return transcript, "captions"
    return None, None


"""RAG ARCHITECTURE"""


def create_database(transcript, video_id):
    docs = Document(
        page_content=transcript,
        metadata={"source": "Youtube", "video_id": video_id}
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Chunk_size,
        chunk_overlap=Chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents([docs])
    embeeding_model = HuggingFaceEmbeddings(
        model_name=Embeed_model,
        model_kwargs={"device": "cpu"}
    )
    persist_dir = f"./chroma_db/{video_id}"
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeeding_model,
        persist_directory=persist_dir,
        collection_name=f"video_{video_id}",
    )
    return vector_store


def load_existing_database(video_id):
    persist_dir = f"./chroma_db/{video_id}"
    if not os.path.exists(persist_dir):
        return None
    try:
        embeedings = HuggingFaceEmbeddings(
            model_name=Embeed_model,
            model_kwargs={"device": "cpu"}
        )
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeedings,
            collection_name=f"video_{video_id}",
        )
        if vector_store._collection.count() > 0:
            return vector_store
        return None
    except Exception:
        return None


def loading_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=Chat_LLM,
        temperature=0.4,
    )


parser = StrOutputParser()


def summerization(transcript):
    llm = loading_llm()
    if len(transcript.split()) < 3000:
        prompt = PromptTemplate.from_template(
            """You are an expert at summarizing YouTube transcripts.
Transcript:
{transcript}

Please provide a comprehensive summary that includes:
1. Main topic and purpose of the video
2. Key points discussed (bullet points)
3. Important conclusions or takeaways
4. Any actionable insights mentioned

Summary:"""
        )
        # len < 3000 summary chain
        chain = prompt | llm | parser
        return chain.invoke({"transcript": transcript})
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=300
        )
        chunks = splitter.split_text(transcript)

        chunk_prompt = PromptTemplate.from_template(
            """Summarize the following section of a video transcript in 3-5 sentences:

{chunk}

Summary:"""
        )
        chunk_chain = chunk_prompt | llm | parser
        summeize_chunks = []
        for i, chunk in enumerate(chunks):
            summary = chunk_chain.invoke({"chunk": chunk})
            summeize_chunks.append(summary)

        combined = "\n\n".join(
            [f"Section {i+1}: {s}" for i, s in enumerate(summeize_chunks)]
        )
        final_prompt = PromptTemplate.from_template(
            """Based on these section summaries from a video, create one comprehensive summary:

{summaries}

Final Summary (include main topic, key points, and takeaways):"""
        )

        final_chain = final_prompt | llm | StrOutputParser()
        return final_chain.invoke({"summaries": combined})


def answer_question(question, vectordb):
    llm = loading_llm()
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_Results, "fetch_k": 10},
    )
    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that answers questions about a YouTube video.
Use ONLY the transcript excerpts below to answer the question.
If the answer is not in the transcript, say "This topic wasn't covered in the video."

Transcript excerpts:
{context}

Question: {question}

Answer (be specific and reference what was said in the video):"""
    )

    def format_docs(docs):
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | parser
    )
    answer = rag_chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:200] + "..." for doc in source_docs],
    }


def validate_video_id(video_id: str) -> bool:
    """Basic validation of YouTube video ID format."""
    # YouTube IDs are 11 characters, alphanumeric + - and _
    pattern = r"^[a-zA-Z0-9_-]{11}$"
    return bool(re.match(pattern, video_id.strip()))


def main():
    # ---- Page Config ----
    st.set_page_config(
        page_title="YouTube Video AI",
        page_icon="🎥",
        layout="wide",
    )

    # ---- Custom CSS ----
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FF0000;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        .info-box {
            background: #f0f7ff;
            border-left: 4px solid #0066cc;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .source-box {
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 0.8rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            color: #555;
            margin: 0.5rem 0;
        }
        .success-box {
            background: #f0fff4;
            border-left: 4px solid #00cc66;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---- Header ----
    st.markdown(
        '<div class="main-header">🎥 YouTube Video AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Summarize any YouTube video or ask questions about its content (using captions only)</div>',
        unsafe_allow_html=True,
    )

    # ---- Initialize Session State ----
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if GROQ_API_KEY == "your_groq_api_key_here":
        st.error("⚠️ Please set your GROQ_API_KEY in the code or as an environment variable.")
        st.info("Get a free API key at: https://console.groq.com")
        st.stop()

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Settings")
        st.markdown("---")

        # Video ID input
        st.subheader("📹 Video Input")
        video_id_input = st.text_input(
            "YouTube Video ID",
            placeholder="e.g. dQw4w9WgXcQ",
            help="The 11-character ID from the YouTube URL: youtube.com/watch?v=**VIDEO_ID**",
        )

        # Helper to extract video ID from full URL
        st.caption("Or paste full URL:")
        full_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
        )
        if full_url:
            # Extract video ID from URL
            match = re.search(
                r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
                full_url,
            )
            if match:
                video_id_input = match.group(1)
                st.success(f"Video ID: `{video_id_input}`")

        st.markdown("---")
        process_btn = st.button(
            "🚀 Process Video",
            type="primary",
            use_container_width=True,
        )

        # Show current video info
        if st.session_state.video_id:
            st.markdown("---")
            st.success(f"✅ Loaded: `{st.session_state.video_id}`")
            if st.session_state.transcript:
                word_count = len(st.session_state.transcript.split())
                st.caption(f"📝 Transcript: {word_count:,} words")

            # Clear button
            if st.button(
                "🗑️ Clear & Load New Video",
                use_container_width=True,
            ):
                st.session_state.vector_store = None
                st.session_state.transcript = None
                st.session_state.video_id = None
                st.session_state.chat_history = []
                st.rerun()

    # ---- Process Video ----
    if process_btn:
        if not video_id_input:
            st.error("Please enter a YouTube Video ID or URL")
        elif not validate_video_id(video_id_input.strip()):
            st.error(
                "Invalid Video ID format. YouTube IDs are 11 characters (letters, numbers, - and _)"
            )
        else:
            video_id = video_id_input.strip()

            # Check if already processed
            existing_store = load_existing_database(video_id)

            if existing_store and video_id == st.session_state.video_id:
                st.info("✅ This video is already loaded!")

            elif existing_store:
                # Video processed before — load from disk
                st.session_state.vector_store = existing_store
                st.session_state.video_id = video_id
                st.session_state.chat_history = []
                st.success("✅ Loaded from saved index (no re-processing needed)")

            else:
                # New video — process from captions only
                with st.spinner("🔍 Fetching YouTube captions..."):
                    transcript, source_method = load_trascript(video_id)

                if not transcript:
                    st.error(
                        "No captions available for this video. "
                        "This app now uses only YouTube captions and does not transcribe audio."
                    )
                    st.stop()

                with st.spinner(
                    "🧠 Embedding transcript and building vector store..."
                ):
                    try:
                        vector_store = create_database(transcript, video_id)
                        st.session_state.vector_store = vector_store
                        st.session_state.transcript = transcript
                        st.session_state.video_id = video_id
                        st.session_state.chat_history = []

                        word_count = len(transcript.split())
                        st.success(
                            f"✅ Ready! Transcript: {word_count:,} words | Source: {source_method}"
                        )

                    except Exception as e:
                        st.error(f"Failed to create vector store: {str(e)}")
                        st.stop()

    if st.session_state.vector_store is None:
        # Welcome screen
        st.markdown(
            """
        <div class="info-box">
        <h3>👋 How to use this app:</h3>
        <ol>
            <li>Find a YouTube video you want to analyze</li>
            <li>Copy the Video ID from the URL (e.g., <code>dQw4w9WgXcQ</code> from <code>youtube.com/watch?v=dQw4w9WgXcQ</code>)</li>
            <li>Paste it in the sidebar and click <strong>Process Video</strong></li>
            <li>Choose to summarize the video or ask any question about it</li>
        </ol>
        <p><strong>Note:</strong> This app uses only the video's captions via YouTubeTranscriptApi.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.subheader("💡 Try these example Video IDs:")
        col1, col2 = st.columns(2)
        with col1:
            st.code("3blue1brown_attention_id")
            st.caption("3Blue1Brown: Attention in Transformers")
        with col2:
            st.code("ySus5ZS0b94")
            st.caption("Andrew Ng: What is Machine Learning")

    else:
        tab1, tab2, tab3 = st.tabs(
            ["📋 Summarize", "💬 Ask Questions", "📄 View Transcript"]
        )
        with tab1:
            st.subheader("📋 Video Summary")
            st.caption("Get a comprehensive summary of the entire video")

            if st.button("✨ Generate Summary", type="primary"):
                if st.session_state.transcript:
                    with st.spinner("🤔 Generating summary..."):
                        try:
                            summary = summerization(st.session_state.transcript)
                            st.markdown("### Summary")
                            st.markdown(summary)

                            # Download button
                            st.download_button(
                                label="📥 Download Summary",
                                data=summary,
                                file_name=f"summary_{st.session_state.video_id}.txt",
                                mime="text/plain",
                            )
                        except Exception as e:
                            st.error(f"Summarization failed: {str(e)}")
                else:
                    st.warning(
                        "Transcript not available in session. Please reload the video."
                    )

        with tab2:
            st.subheader("💬 Ask Questions About the Video")
            st.caption(
                "Ask anything — the AI will find the answer in the video transcript (captions)."
            )

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
                    if msg["role"] == "assistant" and "sources" in msg:
                        with st.expander("📚 Source excerpts from transcript"):
                            for i, source in enumerate(msg["sources"]):
                                st.markdown(
                                    f'<div class="source-box"><strong>Excerpt {i+1}:</strong> {source}</div>',
                                    unsafe_allow_html=True,
                                )

            question = st.chat_input("Ask a question about the video...")

            if question:
                st.session_state.chat_history.append(
                    {
                        "role": "user",
                        "content": question,
                    }
                )

                with st.chat_message("user"):
                    st.write(question)

                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching transcript..."):
                        try:
                            result = answer_question(
                                question,
                                st.session_state.vector_store,
                            )

                            st.write(result["answer"])

                            with st.expander("📚 Source excerpts from transcript"):
                                for i, source in enumerate(result["sources"]):
                                    st.markdown(
                                        f'<div class="source-box"><strong>Excerpt {i+1}:</strong> {source}</div>',
                                        unsafe_allow_html=True,
                                    )

                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": result["answer"],
                                    "sources": result["sources"],
                                }
                            )

                        except Exception as e:
                            error_msg = f"Failed to answer: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append(
                                {
                                    "role": "assistant",
                                    "content": error_msg,
                                }
                            )

            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()

        with tab3:
            st.subheader("📄 Full Transcript")

            if st.session_state.transcript:
                word_count = len(st.session_state.transcript.split())
                st.caption(f"Total words: {word_count:,}")

                search_term = st.text_input(
                    "🔍 Search in transcript",
                    placeholder="Type to search...",
                )
                if search_term:
                    highlighted = st.session_state.transcript.replace(
                        search_term,
                        f"**{search_term}**",
                    )
                    st.markdown(highlighted)
                else:
                    st.text_area(
                        "Transcript",
                        st.session_state.transcript,
                        height=400,
                    )

                st.download_button(
                    label="📥 Download Transcript",
                    data=st.session_state.transcript,
                    file_name=f"transcript_{st.session_state.video_id}.txt",
                    mime="text/plain",
                )
            else:
                st.info(
                    "Transcript not available in current session. The video is still loaded for Q&A."
                )

