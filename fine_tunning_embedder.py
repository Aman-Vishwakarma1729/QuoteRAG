from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os
import torch
import ast 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


processed_data = os.path.join(os.getcwd(), 'dataset', 'processed_quotes_data_1.csv')
df = pd.read_csv(processed_data)

df = df.dropna(subset=['quote', 'tags'])
df = df[df['quote'].apply(lambda x: isinstance(x, str))]
df = df[df['tags'].apply(lambda x: isinstance(x, str) and x.startswith('['))]
train_examples = []
for i, row in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:
            quote1 = row['quote']
            quote2 = row2['quote']

            try:
                tags1 = ast.literal_eval(row['tags'])
                tags2 = ast.literal_eval(row2['tags'])
            except (ValueError, SyntaxError):
                continue  

            label = 1.0 if set(tags1) & set(tags2) else 0.0
            train_examples.append(InputExample(texts=[quote1, quote2], label=label))

model = SentenceTransformer('all-MiniLM-L6-v2')
model.to(device)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

model_path = os.path.join(os.getcwd(), 'model_artifacts', 'finetuned_sentence_transformer')
os.makedirs(model_path, exist_ok=True)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path=model_path,
    use_amp=True 
)

print(f"Model fine-tuned and saved to: {model_path}")
