from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
import ast 
import random
import gc
from tqdm import tqdm
import logging

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.75)
    
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

processed_data = os.path.join(os.getcwd(), 'dataset', 'processed_quotes_data_1.csv')
df = pd.read_csv(processed_data)

print(f"Initial dataset size: {len(df)}")

df = df.dropna(subset=['quote', 'tags'])
df = df[df['quote'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 10)]
df = df[df['tags'].apply(lambda x: isinstance(x, str) and x.startswith('['))]

print(f"Cleaned dataset size: {len(df)}")

valid_rows = []
for idx, row in df.iterrows():
    try:
        tags = ast.literal_eval(row['tags'])
        if isinstance(tags, list) and len(tags) > 0:
            valid_rows.append({
                'quote': row['quote'],
                'tags': tags
            })
    except (ValueError, SyntaxError):
        continue

df_clean = pd.DataFrame(valid_rows)
print(f"Final dataset size: {len(df_clean)}")

def generate_smart_examples(df, max_examples=25000):
    """Generate balanced examples with efficient sampling"""
    train_examples = []
    
    data_list = df.to_dict('records')
    
    print("Generating positive examples...")
    positive_examples = []
    target_positive = max_examples // 2
    
    tag_quotes = {}
    for i, row in enumerate(data_list):
        for tag in row['tags']:
            if tag not in tag_quotes:
                tag_quotes[tag] = []
            tag_quotes[tag].append(i)
    
    for tag, indices in tag_quotes.items():
        if len(indices) < 2:
            continue
            
        sample_size = min(50, len(indices) * (len(indices) - 1) // 4)
        for _ in range(sample_size):
            if len(positive_examples) >= target_positive:
                break
                
            i, j = random.sample(indices, 2)
            quote1 = data_list[i]['quote']
            quote2 = data_list[j]['quote']
            positive_examples.append(InputExample(texts=[quote1, quote2], label=1.0))
        
        if len(positive_examples) >= target_positive:
            break
    
    print("Generating negative examples...")
    negative_examples = []
    target_negative = max_examples - len(positive_examples)
    
    attempts = 0
    max_attempts = target_negative * 3
    
    while len(negative_examples) < target_negative and attempts < max_attempts:
        attempts += 1
        
        i, j = random.sample(range(len(data_list)), 2)
        row1, row2 = data_list[i], data_list[j]
        
    
        if not (set(row1['tags']) & set(row2['tags'])):
            negative_examples.append(InputExample(
                texts=[row1['quote'], row2['quote']], 
                label=0.0
            ))
    
    all_examples = positive_examples + negative_examples
    random.shuffle(all_examples)
    
    print(f"Generated {len(positive_examples)} positive and {len(negative_examples)} negative examples")
    return all_examples

train_examples = generate_smart_examples(df_clean)

del df, df_clean
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

batch_size = 8 if device == "cuda" else 4
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)

model_path = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer')
os.makedirs(model_path, exist_ok=True)

print(f"Starting training...")
print(f"Total examples: {len(train_examples)}")
print(f"Batch size: {batch_size}")
print(f"Total batches: {len(train_dataloader)}")

epochs = 3
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    try:
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=50,
            use_amp=True,
            show_progress_bar=True
        )
   
        epoch_path = os.path.join(model_path, f'checkpoint-epoch-{epoch+1}')
        model.save(epoch_path)
        print(f"Saved checkpoint: {epoch_path}")

        if device == "cuda":
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated() / 1e9
            print(f"GPU memory usage: {current_memory:.2f} GB")
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM error in epoch {epoch + 1}. Reducing batch size...")
            
            del train_dataloader
            gc.collect()
            torch.cuda.empty_cache()
            
            batch_size = max(2, batch_size // 2)
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            print(f"Retrying with batch size: {batch_size}")
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=25,
                use_amp=True
            )
        else:
            raise e

final_path = os.path.join(model_path, 'final_model')
model.save(final_path)
print(f"\nTraining completed! Final model saved to: {final_path}")

if device == "cuda":
    torch.cuda.empty_cache()
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")