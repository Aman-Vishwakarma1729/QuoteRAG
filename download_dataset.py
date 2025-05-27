import pandas as pd
from datasets import load_dataset
import os

dataset = load_dataset("Abirate/english_quotes")

df = pd.DataFrame(dataset['train'])

dataset_path = os.path.join(os.getcwd(),'dataset','raw_quotes_data.csv')
df.to_csv(dataset_path, index=False)
