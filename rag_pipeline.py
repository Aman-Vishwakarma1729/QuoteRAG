import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import os
import ast

model_path = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer', 'final_model')

if not os.path.exists(model_path):
    print(f"Final model not found at {model_path}")
    
    checkpoint_dir = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer')
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-epoch-')]
    
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
        model_path = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Using latest checkpoint: {model_path}")
    else:
        print("No fine-tuned model found, using base model")
        model_path = 'all-MiniLM-L6-v2'

print(f"Loading model from: {model_path}")

try:
    model = SentenceTransformer(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to base model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

processed_data = os.path.join(os.getcwd(), 'dataset', 'processed_quotes_data_1.csv')
df = pd.read_csv(processed_data)

df = df.dropna(subset=['quote', 'tags'])
df = df[df['quote'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 10)]
df = df[df['tags'].apply(lambda x: isinstance(x, str) and x.startswith('['))]

def parse_tags(tags_str):
    try:
        tags = ast.literal_eval(tags_str)
        return tags if isinstance(tags, list) else []
    except:
        return []

df['parsed_tags'] = df['tags'].apply(parse_tags)
df = df[df['parsed_tags'].apply(len) > 0]

print(f"Loaded {len(df)} quotes for indexing")

quotes = df['quote'].tolist()

index_path = 'quotes_index.faiss'
embeddings_path = 'quotes_embeddings.npy'

if os.path.exists(index_path) and os.path.exists(embeddings_path):
    print("Loading existing FAISS index...")
    index = faiss.read_index(index_path)
    embeddings = np.load(embeddings_path)
    print(f"Loaded index with {index.ntotal} embeddings")
else:
    print("Creating new embeddings and FAISS index...")
    batch_size = 100
    embeddings_list = []
    
    for i in range(0, len(quotes), batch_size):
        batch = quotes[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        embeddings_list.append(batch_embeddings)
        print(f"Processed batch {i//batch_size + 1}/{(len(quotes)-1)//batch_size + 1}")
    
    embeddings = np.vstack(embeddings_list)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    np.save(embeddings_path, embeddings)
    print(f"Created and saved FAISS index with {index.ntotal} embeddings")

print("Loading LLM...")
try:
    llm = pipeline('text-generation', 
                   model='microsoft/DialoGPT-medium', 
                   device=0 if os.system('nvidia-smi') == 0 else -1,
                   max_length=512)
    print("Loaded DialoGPT-medium")
except Exception as e:
    print(f"Error loading DialoGPT: {e}")
    print("Falling back to GPT-2...")
    llm = pipeline('text-generation', 
                   model='gpt2', 
                   device=-1,
                   max_length=512)

def retrieve_and_generate(query, k=5):

    print(f"Processing query: {query}")
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    distances, indices = index.search(query_embedding, k)

    retrieved_df = df.iloc[indices[0]]
    retrieved = []
    
    for _, row in retrieved_df.iterrows():
        retrieved.append({
            'quote': row['quote'],
            'author': row.get('author', 'Unknown'),
            'tags': row['parsed_tags'],
            'original_tags': row['tags']
        })
    
    context_parts = []
    for i, r in enumerate(retrieved):
        context_parts.append(f"{i+1}. Quote: \"{r['quote']}\"")
        context_parts.append(f"   Author: {r['author']}")
        context_parts.append(f"   Tags: {', '.join(r['tags'])}")
        context_parts.append("")
    
    context = "\n".join(context_parts)

    prompt = f"""Based on the following quotes, answer the user's query.

Query: {query}

Relevant Quotes:
{context}

Please provide a helpful response that references the most relevant quotes and their authors."""

    print("Generating response...")
    
    try:

        response = llm(prompt, 
                      max_new_tokens=200,
                      do_sample=True,
                      temperature=0.7,
                      pad_token_id=llm.tokenizer.eos_token_id)
        
        generated_text = response[0]['generated_text']
        
        if prompt in generated_text:
            answer = generated_text.replace(prompt, "").strip()
        else:
            answer = generated_text
            
    except Exception as e:
        print(f"Error generating response: {e}")
        answer = f"I found {len(retrieved)} relevant quotes about your query. Here are the most relevant ones:\n\n"
        for i, r in enumerate(retrieved[:3]):
            answer += f"{i+1}. \"{r['quote']}\" - {r['author']}\n"
    
    return {
        "query": query,
        "answer": answer,
        "retrieved_quotes": retrieved,
        "distances": distances[0].tolist(),
        "num_retrieved": len(retrieved)
    }

def search_quotes_by_author(author_name, k=10):

    author_quotes = df[df['author'].str.contains(author_name, case=False, na=False)]
    
    if len(author_quotes) == 0:
        return {"error": f"No quotes found for author: {author_name}"}
    
    result_quotes = []
    for _, row in author_quotes.head(k).iterrows():
        result_quotes.append({
            'quote': row['quote'],
            'author': row['author'],
            'tags': row['parsed_tags']
        })
    
    return {
        "author": author_name,
        "quotes_found": len(author_quotes),
        "quotes": result_quotes
    }

def search_quotes_by_tag(tag_name, k=10):

    tag_quotes = df[df['parsed_tags'].apply(lambda tags: any(tag_name.lower() in tag.lower() for tag in tags))]
    
    if len(tag_quotes) == 0:
        return {"error": f"No quotes found for tag: {tag_name}"}
    
    result_quotes = []
    for _, row in tag_quotes.head(k).iterrows():
        result_quotes.append({
            'quote': row['quote'],
            'author': row['author'],
            'tags': row['parsed_tags']
        })
    
    return {
        "tag": tag_name,
        "quotes_found": len(tag_quotes),
        "quotes": result_quotes
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Quote RAG System Ready!")
    print("="*50)
    
    
    test_queries = [
        "Quotes about hope and perseverance",
        "What did Oscar Wilde say about life?",
        "Inspirational quotes about success"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        print("-" * 30)
        
        result = retrieve_and_generate(query, k=3)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {result['num_retrieved']} quotes")
        
        print("\nTop retrieved quotes:")
        for i, quote in enumerate(result['retrieved_quotes'][:2]):
            print(f"{i+1}. \"{quote['quote'][:100]}...\" - {quote['author']}")
        
        print("\n" + "="*50)
    

    print("\nEntering interactive mode. Type 'quit' to exit.")
    while True:
        user_query = input("\nEnter your query: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if user_query:
            result = retrieve_and_generate(user_query)
            print(f"\nAnswer: {result['answer']}")
            
            print(f"\nRelevant quotes ({result['num_retrieved']} found):")
            for i, quote in enumerate(result['retrieved_quotes'][:3]):
                print(f"{i+1}. \"{quote['quote']}\" - {quote['author']}")
                print(f"   Tags: {', '.join(quote['tags'][:5])}")  
                print()
    
    print("Goodbye!")