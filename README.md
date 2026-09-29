# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot that answers questions using information retrieved from a medical document.

## Features

- PDF document processing and text chunking
- Semantic search using Hugging Face embeddings
- FAISS vector database for similarity search
- Qwen LLM for answer generation
- Streamlit-based chat interface
- Displays retrieved source documents

## Tech Stack

Python | LangChain | Hugging Face | Qwen | FAISS | Sentence Transformers | Streamlit

## How It Works

```text
Medical PDF
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS
    ↓
User Query
    ↓
Similarity Search
    ↓
Relevant Context
    ↓
Qwen
    ↓
Answer
```

## Setup

```bash
pipenv install
pipenv run python create_memory_for_llm.py
pipenv run streamlit run app.py
```

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
```

The medical PDF and FAISS vector database are excluded from the repository.