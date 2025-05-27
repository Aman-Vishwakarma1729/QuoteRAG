from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from datasets import Dataset
import pandas as pd
import json
from rag_pipeline import retrieve_and_generate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Configure open-source LLM for RAGAS
def setup_open_source_llm():
    try:
        # Use the same model as in rag_pipeline.py (DialoGPT-medium or fallback to gpt2)
        model_name = 'microsoft/DialoGPT-medium'
        device = 0 if torch.cuda.is_available() else -1
        print(f"Setting up LLM: {model_name} on device {device}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad_token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create pipeline
        llm_pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        # Wrap in LangChain's HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=llm_pipeline)
        return llm
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to gpt2...")
        model_name = 'gpt2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm_pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            device=-1,
            max_length=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)

# Example evaluation queries (same as provided)
eval_queries = [
    "Quotes about insanity attributed to Einstein",
    "Motivational quotes tagged 'accomplishment'",
    "All Oscar Wilde quotes with humor"
]

# Ground truth answers for evaluation (manually curated for better recall/correctness)
ground_truths = [
    "The definition of insanity is doing the same thing over and over again and expecting different results. - Albert Einstein",
    "Success is not the absence of obstacles, but the courage to push through them. - Unknown\nAccomplishment is the result of perseverance. - Unknown",
    "Many lack the originality to lack originality. - Oscar Wilde\nMany lack originality. - Oscar Wilde"
]

# Generate responses
print("Generating responses for evaluation queries...")
responses = []
for q in eval_queries:
    try:
        result = retrieve_and_generate(q, k=5)
        responses.append(result)
    except Exception as e:
        print(f"Error processing query '{q}': {e}")
        responses.append({"answer": "", "retrieved_quotes": [], "distances": [], "num_retrieved": 0})

# Prepare dataset for RAGAS
eval_data = {
    "question": eval_queries,
    "answer": [r['answer'] for r in responses],
    "contexts": [[f"Quote: {c['quote']}\nAuthor: {c['author']}\nTags: {', '.join(c['tags'])}" for c in r['retrieved_quotes']] for r in responses],
    "ground_truth": ground_truths
}

# Convert to HuggingFace Dataset
dataset = Dataset.from_dict(eval_data)

# Set up open-source LLM for RAGAS
llm = setup_open_source_llm()

# Evaluate with RAGAS
metrics = [
    faithfulness,          # Measures if the answer is consistent with the context
    answer_relevancy,     # Measures how relevant the answer is to the question
    context_precision,    # Measures if retrieved contexts are relevant
    context_recall,       # Measures if all relevant quotes are retrieved
    answer_correctness    # Measures factual correctness against ground truth
]

print("Running RAGAS evaluation...")
try:
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,  # Use open-source LLM
        raise_exceptions=False
    )
except Exception as e:
    print(f"Evaluation failed: {e}")
    result = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0, "answer_correctness": 0.0}

# Convert results to a dictionary for detailed output
results_dict = {
    "aggregate_scores": result,
    "per_query_results": []
}

# Add per-query details
for i, (query, response, contexts) in enumerate(zip(eval_queries, responses, eval_data['contexts'])):
    per_query = {
        "query": query,
        "answer": response['answer'],
        "retrieved_quotes": [
            {"quote": c['quote'], "author": c['author'], "tags": c['tags'], "distance": d}
            for c, d in zip(response['retrieved_quotes'], response['distances'])
        ],
        "contexts": contexts
    }
    results_dict['per_query_results'].append(per_query)

# Save results
with open('evaluation_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Print summary
print("\nEvaluation Results:")
print(f"Faithfulness: {result['faithfulness']:.4f}")
print(f"Answer Relevancy: {result['answer_relevancy']:.4f}")
print(f"Context Precision: {result['context_precision']:.4f}")
print(f"Context Recall: {result['context_recall']:.4f}")
print(f"Answer Correctness: {result['answer_correctness']:.4f}")
print("\nDetailed results saved to 'evaluation_results.json'")

# Discussion
print("\nDiscussion:")
print("The RAG pipeline was evaluated using RAGAS with an open-source LLM (DialoGPT-medium or gpt2).")
print("- Faithfulness: Checks if the generated answer aligns with the retrieved quotes.")
print("- Answer Relevancy: Ensures the answer addresses the query directly.")
print("- Context Precision: Evaluates if the top retrieved quotes are relevant.")
print("- Context Recall: Verifies if all relevant quotes were retrieved (requires ground truth).")
print("- Answer Correctness: Compares the answer to the ground truth.")
print("Challenges: The open-source LLM may have lower reasoning capabilities compared to OpenAI models, potentially affecting metric scores. Retrieval quality depends on the fine-tuned Sentence-Transformer; suboptimal embeddings may lower context precision/recall.")