
import streamlit as st
import os 
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain


load_dotenv()

## Load the NVIDIA API Key
os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")
llm=ChatNVIDIA(model="meta/llama3-70b-instruct")
def vectore_embedding():
    if "vectore" not in st.session_state:
        st.session_state.embeddings=NVIDIAEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./us_census/data")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        st.session_state.final_document=st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        st.session_state.vectore=FAISS.from_documents(st.session_state.final_document,st.session_state.embeddings)


st.title("NVIDIA NIM Demo")
prompt=ChatPromptTemplate.from_template(
    """   
Anser the question based on given context only.
Please provide the moste accurate response based on the question
<context>
{context}
<context>
Question:{input}
"""
)

prompt1=st.text_input("Enter a your question from Documents")

if st.button("Document Embedding"):
    vectore_embedding()
    st.write("Faiss vectore store DB is ready Using Nvidia Embeddings")

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectore.as_retriever()
    retriver_chin=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retriver_chin.invoke({'input':prompt1})
    print("Response time:",time.process_time()-start)
    st.write(response['answer'])


    # with a streamlit expanded
    with st.expander("Document Similarity Search"):
        # find the relevent chunk
        for i ,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------")