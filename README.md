<h1 align="center">QuoteRAG: Intelligent Quote Retrieval and Generation System</h1>
<div align="center">
  <img src="title.png" alt="Designer" width="500"/>
</div>

## Introduction

QuoteRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to intelligently search and generate contextual responses from a curated dataset of quotes. By combining semantic search with natural language generation, QuoteRAG delivers relevant quotes and thoughtful responses to user queries about topics, authors, or themes. Built with modern LLMs and a user-friendly Streamlit interface, this project showcases expertise in natural language processing (NLP), embeddings, vector search, and web application development.

The system leverages a fine-tuned SentenceTransformer model for semantic quote retrieval, FAISS for efficient vector search, and an open-source large language model (LLM) like DialoGPT-medium or GPT-2 for response generation. The project also includes data preprocessing, model fine-tuning, RAG pipeline evaluation, and a visually appealing Streamlit application, demonstrating a full-stack approach to building AI-driven applications.

## Project Overview

QuoteRAG is designed to:

- **Retrieve Relevant Quotes**: Use a fine-tuned SentenceTransformer model and FAISS to perform semantic search on a quote dataset.
- **Generate Contextual Responses**: Integrate retrieved quotes into natural language responses using an LLM.
- **Provide an Interactive Interface**: Offer a Streamlit-based web application for seamless user interaction, including smart search, author/tag-based search, and analytics.
- **Evaluate Performance**: Use the RAGAS framework to assess the quality of the RAG pipeline, measuring metrics like faithfulness, answer relevancy, and context precision/recall.

This project highlights skills in:

- **Machine Learning & NLP**: Fine-tuning embedding models, implementing RAG pipelines, and evaluating performance with RAGAS.
- **Data Processing**: Cleaning and preprocessing quote datasets for robust model training and retrieval.
- **Software Development**: Building a scalable, modular codebase with Python, Pandas, and Streamlit.
- **GPU Optimization**: Efficient memory management for model training and inference.
- **Visualization**: Creating interactive data visualizations with Plotly for user insights.

## Project Structure

The project is organized into modular scripts, each handling a specific component of the pipeline:

- `download_dataset.py`: Downloads the raw quote dataset from HuggingFace and saves it as `raw_quotes_data.csv`.
- `data_preprocessing_and_analysis.ipynb`: Preprocesses the raw dataset, performing cleaning and analysis, and saves the processed data as `processed_quotes_data_1.csv`.
- `fine_tuning_embedder.py`: Fine-tunes a SentenceTransformer model with a baseline approach for semantic embeddings.
- `fine_tuning_embedder_1.py`: An optimized version of the fine-tuning script with improved memory management and GPU utilization.
- `rag_pipeline.py`: Implements the core RAG pipeline, combining FAISS-based retrieval with LLM-based response generation.
- `rag_evaluation.py`: Evaluates the RAG pipeline using RAGAS metrics like faithfulness, answer relevancy, context precision, context recall, and answer correctness.
- `quoterag_app.py`: A Streamlit web application providing an interactive interface for quote search, author/tag filtering, and analytics visualization.

## Setup Instructions

To set up and run QuoteRAG, follow these steps:

### Prerequisites

- Python 3.10
- Conda (for virtual environment management)
- NVIDIA GPU (optional, for faster training and inference)
- Required libraries: `pandas`, `sentence-transformers`, `transformers`, `faiss-cpu` (or `faiss-gpu`), `torch`, `streamlit`, `plotly`, `ragas`, `datasets`, `langchain-huggingface`

### Installation

1. **Create and Activate Conda Environment**

   ```bash
   conda create -p quoterag_venv python==3.10
   conda activate quoterag_venv
   ```

2. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install -r requirement.txt
   ```

   For GPU support, replace `faiss-cpu` with `faiss-gpu`.

3. **Download the Dataset**

   Run the dataset download script to fetch the raw quote dataset:

   ```bash
   python download_dataset.py
   ```

   This saves the dataset as `dataset/raw_quotes_data.csv`.

4. **Preprocess the Data**

   Execute the preprocessing notebook to clean and analyze the dataset:

   ```bash
   jupyter notebook data_preprocessing_and_analysis.ipynb
   ```

   This generates `dataset/processed_quotes_data_1.csv`.

5. **Fine-Tune the SentenceTransformer Model**

   Run the optimized fine-tuning script for the embedding model:

   ```bash
   python fine_tuning_embedder_1.py
   ```

   This creates a fine-tuned model in `model_artifacts/finetuned_sentence_transformer/final_model`. The script includes memory-efficient training with dynamic batch size adjustment and GPU optimization.

6. **Run the RAG Pipeline**

   Test the RAG pipeline for quote retrieval and response generation:

   ```bash
   python rag_pipeline.py
   ```

   This loads the fine-tuned model, creates a FAISS index, and supports interactive queries.

7. **Evaluate the RAG Pipeline**

   Assess the pipeline's performance using RAGAS:

   ```bash
   python rag_evaluation.py
   ```

   Results are saved to `evaluation_results.json`, including metrics like faithfulness and context precision.

8. **Launch the Streamlit Application**

   Start the web interface:

   ```bash
   streamlit run quoterag_app.py
   ```

   Access the application at `http://localhost:8501` for interactive quote search and analytics.

## Usage

### Command-Line Interface

Run `rag_pipeline.py` for an interactive command-line experience:

```bash
python rag_pipeline.py
```

Enter queries like "Quotes about hope" or "Oscar Wilde quotes about life". Type `quit` to exit.

### Streamlit Web Application

The Streamlit app (`quoterag_app.py`) offers:

- **Smart Search**: Enter queries like "inspirational quotes about perseverance" to retrieve and generate contextual responses.
- **Author Search**: Filter quotes by author (e.g., "Oscar Wilde").
- **Tag Search**: Explore quotes by tags (e.g., "motivation").
- **Analytics**: Visualize tag frequencies, quote length distributions, and top authors using Plotly.

The app features a modern UI with custom CSS, quote cards, and interactive elements like like/copy buttons.

### Example Queries

- "Quotes about insanity attributed to Einstein"
- "Motivational quotes tagged 'accomplishment'"
- "All Oscar Wilde quotes with humor"

## Technical Highlights

This project demonstrates advanced technical skills in:

- **NLP and Embeddings**: Fine-tuning a SentenceTransformer model (`all-MiniLM-L6-v2`) using cosine similarity loss and optimized data sampling for positive/negative pairs, as shown in `fine_tuning_embedder_1.py`.
- **Vector Search**: Implementing FAISS for efficient similarity search, with persistent indexing for faster loading (`rag_pipeline.py`).
- **RAG Architecture**: Combining retrieval (FAISS + SentenceTransformer) with generation (DialoGPT-medium or GPT-2) to create a robust RAG pipeline.
- **Evaluation**: Using RAGAS to compute metrics like faithfulness, answer relevancy, context precision, context recall, and answer correctness, showcasing a rigorous approach to model evaluation (`rag_evaluation.py`).
- **Optimization**: Implementing memory-efficient training with dynamic batch size adjustment and GPU memory management (`fine_tuning_embedder_1.py`).
- **Web Development**: Building a responsive Streamlit app with custom CSS, Plotly visualizations, and caching for performance (`quoterag_app.py`).
- **Data Engineering**: Cleaning and preprocessing quote datasets with Pandas, handling missing data, and parsing tags with `ast` (`download_dataset.py`, `data_preprocessing_and_analysis.ipynb`).
- **Robustness**: Fallback mechanisms (e.g., switching to GPT-2 if DialoGPT fails) ensure system reliability across hardware constraints.

## Challenges and Solutions

- **Memory Constraints**: Addressed in `fine_tuning_embedder_1.py` by using batch processing, dynamic batch size reduction, and memory cleanup with `torch.cuda.empty_cache()`.
- **Model Limitations**: Open-source LLMs (DialoGPT-medium, GPT-2) were used due to the lack of access to proprietary models like OpenAI's, with RAGAS evaluation adapted for these models.
- **Dataset Quality**: Ensured robust preprocessing to filter invalid quotes and tags, improving embedding quality and retrieval accuracy.
- **User Experience**: Designed an intuitive Streamlit UI with search tips, analytics, and interactive features to enhance usability.

## Future Improvements

- Integrate more advanced LLMs (e.g., LLaMA or Mistral) for improved response generation.
- Enhance the dataset with additional sources for broader coverage.
- Implement real-time dataset updates and dynamic index rebuilding.
- Add user authentication and personalized quote saving in the Streamlit app.


## Acknowledgments

- **Dataset**: Abirate/english_quotes
