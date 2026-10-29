#=====================================================================
#import all the required libraries
#=====================================================================
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

#=====================================================================
#Page Config
#====================================================================
st.title("C++ RAG with Ollama LLM")

#=====================================================================
#Load & Split Documents
#====================================================================
@st.cache_resource  #@st.cache_resource is a decorator provided by Streamlit that caches the output of the decorated function. When the function is called with the same input parameters, Streamlit will return the cached result instead of executing the function again. This can significantly improve performance, especially for resource-intensive operations like loading and processing documents. In this case, it is used to cache the vectorstore created from the C++ documents, so that it doesn't need to be recreated every time the user interacts with the app.
def load_vectorstore():

    loader = TextLoader('CPP.txt',encoding="utf-8")
    documents = loader.load()
#=====================================================================
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
        )                                                  # splitter is used to split the documents into smaller chunks of text. The chunk_size parameter specifies the maximum size of each chunk, and the chunk_overlap parameter specifies how much overlap there should be between consecutive chunks. This is useful for ensuring that important context is not lost when splitting the documents.
    final_docs = text_splitter.split_documents(documents)
#=====================================================================
    # Create embeddings and store them in a FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(final_docs, embeddings)

    return db

#=====================================================================
#Load Vectorstore
#====================================================================
db = load_vectorstore()
#=====================================================================
#Load Ollama LLM
#====================================================================
llm = Ollama(model="gemma2:2b")

#=====================================================================
#Chat Interface
#=====================================================================
user_question = st.text_input("Ask a question about C++:")

if user_question:
    with st.spinner("Thinking....."):
        # Retrieve relevant documents from the vectorstore
        docs = db.similarity_search(user_question, k=5)
        context = "\n".join([doc.page_content for doc in docs]) #.join() is used to join the content of the retrieved documents into a single string, separated by newlines. This context will be used to answer the user's question.

        prompt = f"""
        Answer the question based on the context below. If the question can't be answered based on the context, say "I don't know".

        Context:
        {context}
        Question:
        {user_question}
        Answer:"""

        # Generate a response using the LLM with the retrieved context
        response = llm.invoke(prompt) # here ivoke is used to call the LLM with the prompt and get the response
        st.subheader("Answer:")
        st.write(response)
