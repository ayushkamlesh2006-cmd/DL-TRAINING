import io
import streamlit as st

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.caption(
    "Upload your documents, ask questions, "
    "and receive answers based on semantic similarity search."
)


# --------------------------------------------------
# 2. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    model_name = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2"]
    )

    ollama_model = st.selectbox(
        "Language Model",
        ["gemma2:2b"]
    )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1500,
        value=800,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=100,
        step=50
    )

    top_k = st.slider(
        "Relevant Chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.info(
        "Embeddings retrieve relevant text. "
        "Gemma generates the answer using that text."
    )


# --------------------------------------------------
# 3. INITIALIZE EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embeddings(name):
    return HuggingFaceEmbeddings(
        model_name=name
    )


# --------------------------------------------------
# 4. INITIALIZE OLLAMA MODEL
# --------------------------------------------------

@st.cache_resource
def load_llm(name):
    return OllamaLLM(
        model=name,
        temperature=0
    )


# --------------------------------------------------
# 5. LOAD DOCUMENTS
# --------------------------------------------------

def load_uploaded_documents(uploaded_files):

    documents = []

    for uploaded_file in uploaded_files:

        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        try:

            # PDF FILE
            if file_name.lower().endswith(".pdf"):

                pdf_reader = PdfReader(
                    io.BytesIO(file_bytes)
                )

                for page_number, page in enumerate(
                    pdf_reader.pages,
                    start=1
                ):

                    page_text = page.extract_text() or ""

                    if page_text.strip():

                        documents.append(
                            Document(
                                page_content=page_text,
                                metadata={
                                    "source": file_name,
                                    "page": page_number
                                }
                            )
                        )

            # TXT FILE
            elif file_name.lower().endswith(".txt"):

                text = file_bytes.decode("utf-8-sig")

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_name
                            }
                        )
                    )

        except Exception as error:

            st.error(
                f"Could not read {file_name}: {error}"
            )

    return documents


# --------------------------------------------------
# 6. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(documents, size, overlap):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )

    return splitter.split_documents(documents)


# --------------------------------------------------
# 7. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(chunks, embeddings):

    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )


# --------------------------------------------------
# 8. UPLOAD DOCUMENTS
# --------------------------------------------------

st.subheader("📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


# --------------------------------------------------
# 9. PROCESS DOCUMENTS
# --------------------------------------------------

if uploaded_files:

    if st.button(
        "Process Documents",
        type="primary"
    ):

        with st.spinner(
            "Reading documents and creating embeddings..."
        ):

            try:

                raw_documents = load_uploaded_documents(
                    uploaded_files
                )

                if not raw_documents:
                    st.error(
                        "No readable text was found in the uploaded files."
                    )
                    st.stop()

                chunks = split_documents(
                    raw_documents,
                    chunk_size,
                    chunk_overlap
                )

                embeddings = load_embeddings(
                    model_name
                )

                vector_db = create_vector_database(
                    chunks,
                    embeddings
                )

                st.session_state["vector_db"] = vector_db
                st.session_state["chunks"] = chunks

                st.session_state["processed_files"] = [
                    file.name for file in uploaded_files
                ]

                st.session_state["embedding_model"] = model_name
                st.session_state["chunk_size"] = chunk_size
                st.session_state["chunk_overlap"] = chunk_overlap

                st.success(
                    "Documents processed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Document processing failed: {error}"
                )


# --------------------------------------------------
# 10. DISPLAY DATABASE INFORMATION
# --------------------------------------------------

if "vector_db" in st.session_state:

    st.divider()

    st.subheader("📊 Vector Database Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        len(st.session_state["processed_files"])
    )

    col2.metric(
        "Text Chunks",
        len(st.session_state["chunks"])
    )

    col3.metric(
        "FAISS Vectors",
        st.session_state["vector_db"].index.ntotal
    )

    with st.expander("View processed files"):

        for file_name in st.session_state["processed_files"]:
            st.write(f"📄 {file_name}")


# --------------------------------------------------
# 11. QUESTION-ANSWERING INTERFACE
# --------------------------------------------------

if "vector_db" in st.session_state:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the main idea of Attention Is All You Need?"
    )

    ask_button = st.button(
        "Get Answer",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching documents and generating an answer..."
            ):

                try:

                    vector_db = st.session_state["vector_db"]

                    # --------------------------------
                    # SIMILARITY SEARCH
                    # --------------------------------

                    relevant_docs = vector_db.similarity_search(
                        question,
                        k=min(
                            top_k,
                            vector_db.index.ntotal
                        )
                    )

                    # --------------------------------
                    # PREPARE CONTEXT
                    # --------------------------------

                    context_parts = []

                    for doc in relevant_docs:

                        source = doc.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = doc.metadata.get(
                            "page"
                        )

                        source_label = source

                        if page is not None:
                            source_label += f", page {page}"

                        context_parts.append(
                            f"Source: {source_label}\n"
                            f"Content:\n{doc.page_content}"
                        )

                    context = "\n\n".join(
                        context_parts
                    )

                    # --------------------------------
                    # PROMPT TEMPLATE
                    # --------------------------------

                    prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """
                                You are a helpful document question-answering
                                assistant.

                                Answer the user's question using the
                                provided document context.

                                Instructions:
                                1. Answer ONLY using information explicitly supported
                                   by the provided document context.

                                2. Do not use general knowledge, medical knowledge,
                                    assumptions, or external information to complete
                                    a missing answer.

                                3. If the context does not contain sufficient evidence,
                                    respond exactly:
                                    "I couldn't find sufficient information in the
                                    uploaded documents to answer this question reliably."

                                4. Do not infer definitions or meanings that are not
                                    supported by the documents.

                                5. If documents contain conflicting information,
                                    identify the conflict and cite the relevant sources.

                                6. Cite the source filename and page number for
                                    factual claims whenever that metadata is available.

                                Document context:
                                {context}
                                """
                            ),
                            (
                                "human",
                                "{question}"
                            )
                        ]
                    )

                    # --------------------------------
                    # LOAD LLM
                    # --------------------------------

                    llm = load_llm(
                        ollama_model
                    )

                    # --------------------------------
                    # BUILD RAG CHAIN
                    # --------------------------------

                    output_parser = StrOutputParser()

                    rag_chain = (
                        prompt
                        | llm
                        | output_parser
                    )

                    # --------------------------------
                    # GENERATE ANSWER
                    # --------------------------------

                    answer = rag_chain.invoke(
                        {
                            "context": context,
                            "question": question
                        }
                    )

                    st.subheader("📝 Answer")

                    st.markdown(answer)

                    # --------------------------------
                    # SHOW RETRIEVED SOURCES
                    # --------------------------------

                    with st.expander(
                        "🔎 View relevant source passages"
                    ):

                        for index, doc in enumerate(
                            relevant_docs,
                            start=1
                        ):

                            st.markdown(
                                f"**Source {index}**"
                            )

                            st.caption(
                                f"File: {doc.metadata.get('source', 'Unknown')}"
                                + (
                                    f" | Page: {doc.metadata['page']}"
                                    if "page" in doc.metadata
                                    else ""
                                )
                            )

                            st.write(
                                doc.page_content
                            )

                            st.divider()

                except Exception as error:

                    st.error(
                        f"Question answering failed: {error}"
                    )

                    st.info(
                        "Check that Ollama is running and the selected "
                        "model is available."
                    )


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit, LangChain, Hugging Face, "
    "FAISS and Ollama."
)