import logging
import pandas as pd
from transformers import GPT2Tokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)

def prepare_csv_data(csv_path, output_path):
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Identify relevant columns (adjust these based on your use case)
    relevant_columns = list(df.columns)

    # Combine text from relevant columns into a single text column
    df['text'] = df[relevant_columns].astype(str).apply(' '.join, axis=1)

    # Save the processed data
    df[['text']].to_csv(output_path, index=False, header=False)

# Example usage for preparing CSV data
csv_path = 'path/to/your/data.csv'
processed_data_path = 'path/to/your/processed_data.txt'

prepare_csv_data(csv_path, processed_data_path)
