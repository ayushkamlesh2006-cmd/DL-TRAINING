import streamlit as st
#inport os
#from dotenv import load_dotenv
#LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#--------------------------------------------------------------------------
# Page Config
#--------------------------------------------------------------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.write("# C++ RAG Chatbot")
st.write("Ask any question related to C++ .")


#---------------------------------------------------------------------------
#Load Environment Variables
#load_dotenv()
#---------------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    # Load the text file
    loader = TextLoader("cpp.txt", encoding="utf-8")
    documents = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    final_documents = text_splitter.split_documents(documents)

    # Create embeddings using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a FAISS vector store from the embeddings
    db = FAISS.from_documents(final_documents, embeddings)

    return db
#----------------------------------------------------------------------------
db = load_vectorstore()

#----------------------------------------------------------------------------
#User Input and Query

query = st.text_input("Enter your question about C++: ")
if query:
    docs = db.similarity_search(query,k=3)
    for i , doc in enumerate(docs):
        st.markdown(f"**Result {i+1}**")
        st.write(doc.page_content)