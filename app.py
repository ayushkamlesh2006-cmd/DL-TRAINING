#import os 
#from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


#creating my prompt 
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.please respond to the Question asked."),
        ("user", "Question:{question}"),
    ]
)

#streamlit Framework
st.title("Langchain demo with gemma2:2b")
input_text = st.text_input("Enter your question here:")

#ollama model
llm = OllamaLLM(model="gemma2:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))