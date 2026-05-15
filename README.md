# 📄 PDF RAG Chatbot (LangChain + Groq + FAISS)

## Overview
This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions from PDF documents using an LLM. It uses FAISS for vector search, HuggingFace embeddings for semantic representation, and Groq (Llama3) for fast inference.

---

## Tech Stack
- LangChain
- LangChain Community / Core
- Groq LLM (Llama3)
- FAISS (Vector Database)
- HuggingFace Sentence Transformers
- Streamlit
- Python

---

## Features
- Load and process PDF documents
- Chunking with recursive text splitter
- Semantic embeddings using HuggingFace
- Vector search with FAISS
- Retrieval-Augmented QA system
- Streamlit web UI

---

## Project Structure

```

rag-project/
│
├── app.py                # Streamlit application
├── notebooks/           # Jupyter notebooks (experiments)
├── us_census/           # PDF documents
├── requirements.txt
├── .gitignore
└── README.md

````

---

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd rag-project
````

### 2. Create virtual environment

```bash
python -m venv ragenv
```

Activate environment:

**Windows (Git Bash):**

```bash
source ragenv/Scripts/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

---

## Run Jupyter Notebook

```bash
jupyter notebook
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

Make sure `app.py` is in the project root directory.

---

## How It Works

* Load PDFs from `us_census/`
* Split documents into chunks
* Convert chunks into embeddings
* Store embeddings in FAISS index
* Retrieve relevant chunks for a query
* Send context + question to Groq LLM
* Return final answer

---

## Common Issues

### Empty document loading

* Ensure PDFs exist inside `us_census/`
* Ensure correct working directory

### FAISS error (empty embeddings)

* Happens when documents are not loaded or splitting fails

### Streamlit file not found

* Run from project root:

```bash
streamlit run app.py
```

---

## Author

Jubril Adekunle

---
