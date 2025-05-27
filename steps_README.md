## Create conda virtual environment with python version 3.10
```bash
conda create -p quoterag_venv python==3.10
```

## Activate the created virtual environment
```bash
conda activate quoterag_venv
```

## Download RAW dataset in .csv format in "dataset" folder as "raw_quotes_data.csv"
```bash
download_dataset.py
```

## Basic data preprocessing and analysis and saving "raw_quotes_data.csv" as "processed_quotes_data_1.csv" in dataset folder.
```bash
data_preprocessing_and_analysis.ipynb
```

## Finetunning embedder model "sentence-transformer". We have two aprroches:
```bash
fine_tunning_embedder.py
```
* The above tunning code use GPU if avaliable for higghly resource intensive, this was challenge so we have better approach below.

```bash
fine_tunning_embedder_1.py
```
* The above tunning code more optimized for memory management and GPU utilization.

## RAG pipeline
```bash 
rag_pipeline.py
```
* The LLM is used to generate a contextual, natural-language answer to the user's query.
* It doesn't retrieve quotes itself — it summarizes and weaves the top-k similar quotes (from FAISS + SentenceTransformer) into a meaningful response.
* A prompt is dynamically created containing:
* ---- The user's query.
* ---- The top-k most relevant quotes retrieved via semantic search.
* ---- Authors and tags of these quotes.
* This prompt is passed to the LLM to guide its response generation, helping it stay anchored to real, relevant data.
* The LLM generates a textual answer that might explain, summarize, or discuss the topic using the context of the retrieved quotes.
* This helps bridge the gap between retrieval (FAISS) and natural language understanding, making the system feel conversational.
* If DialoGPT-medium (which is more conversational) fails to load (e.g., due to hardware limitations), the code falls back to GPT-2, ensuring continued functionality.

## RAG Evaluation
```bash
rag_evaluation.py
```
* This evaluates our RAG pipeline.
* But for better testing we need acces to OPEN-AI which is bit of challengin right now as it is paid.
* We implement ragas for time being to evaluate our pipeline.

## Streamlit application
```bash
quoterag_app.py
```
* This is an basic streamlit application.