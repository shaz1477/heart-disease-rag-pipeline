# Heart Disease RAG Pipeline with Hybrid Retrieval & Iterative Evaluation

An end-to-end Retrieval-Augmented Generation (RAG) pipeline for answering clinical 
heart disease queries from a PDF knowledge base, built with LangChain, ChromaDB, 
HuggingFace embeddings, and Groq (Llama-3.1-8b).

## Pipeline Overview

PDF → Chunking → Embeddings → ChromaDB → Hybrid Retrieval → Reranking → Multi-Query → RAG Chain → RAGAS Evaluation

## Key Features

- **Hybrid Retrieval** — BM25 (keyword) + Dense (semantic) with weighted ensemble (0.4 / 0.6)
- **Cross-Encoder Reranking** — BAAI/bge-reranker-base reranks top chunks (top_n=3)
- **Multi-Query Retrieval** — LLM generates query variants to improve recall
- **Iterative Tuning** — chunk size, overlap, and k tuned across 3 iterations guided by RAGAS
- **RAGAS Evaluation** — faithfulness, answer relevancy, context precision, context recall

## Results

| Metric | Score |
|---|---|
| Faithfulness | 0.9375 |
| Context Recall | 0.6250 |

Context recall improved from **0.25 → 0.75** across iterations.

## Tech Stack

LangChain · ChromaDB · HuggingFace (BAAI/bge-base-en-v1.5) · Groq (Llama-3.1-8b) · RAGAS · BM25 · PyPDF

## Setup

```bash
pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-groq ragas rank_bm25 pypdf python-dotenv
```

Create a `.env` file:HF_TOKEN=your_huggingface_token
GROQ_API_KEY=your_groq_api_key


Then run the notebook: `heart_disease_rag_pipeline.ipynb`
