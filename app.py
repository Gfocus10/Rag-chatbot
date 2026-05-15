import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 PDF RAG Chatbot (LangChain + Groq + FAISS)")


@st.cache_resource
def build_rag():

    loader = PyPDFDirectoryLoader("./us_census")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    You are an intelligent PDF assistant.

    Answer the question ONLY from the provided context.

    If the answer is not present in the context, say:
    "I could not find the answer in the provided documents."

    Provide concise and accurate answers.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return retrieval_chain


chain = build_rag()


query = st.text_input("Ask your question:")

if query:
    result = chain.invoke({"input": query})
    st.write("### Answer")
    st.write(result["answer"])
