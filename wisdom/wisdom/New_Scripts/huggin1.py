import os
import logging
import random
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any, Tuple
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sklearn.model_selection import train_test_split

# Set up logger
logging.basicConfig(level=logging.DEBUG, format="%(line)d %(message)s")
logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = "path/to/your/csv"
MODEL_PATH = "distilbert-base-cased-distilled-sqa"
MAX_LEN = 384
BATCH_SIZE = 6
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = 'cpu'

class CustomDataset(Dataset):
    def __init__(self, encodings: dict, labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item, self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

def load_and_process_data() -> Tuple[pd.DataFrame, List[List[Any]]]:
    # Load CSV and filter columns if needed
    df = pd.read_csv(DATA_DIR).dropna()
    
    # Process questions and convert text into input ids, attention masks, etc.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    encoded_dict = tokenizer(df["description"].tolist(), truncation=True, padding='max_length', max_length=MAX_LEN)
    input_ids = encoded_dict['input_ids']
    attention_masks = encoded_dict['attention_mask']
    dataset = CustomDataset(encoded_dict, df["status"].apply(lambda x: ["open", "closed"].index(x)).tolist())
    
    # Split datasets into train and validation sets
    train_dataset, valid_dataset = train_test_split(dataset, test_size=0.1, random_state=random.randint(1, 10))

    return df, (train_dataset, valid_dataset)

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH).to(device)

    # ... continue with defining optimizers, loss functions, loop through epochs, saving best model, etc.

def answer_question(context: str, question: str) -> str:
    # Tokenize context and question
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokens = tokenizer([context], return_tensors="pt").to(DEVICE)
    tokens_q = tokenizer([question], return_tensors="pt").to(DEVICE)

    # Predict start and end positions of the answer
    start_positions, end_positions = model(**tokens, **tokens_q)[1][:, :, :].to("cpu").detach().numpy()

    # Get actual string indexes
    start_index = np.argmax(start_positions)
    end_index = np.argmax(end_positions) + 1  # Add one because of how slicing works

    # Extract the predicted answer
    answer = context[start_index:end_index]
    return answer

if __name__ == "__main__":
    logger.debug("Starting...")
    data = load_and_process_data()
    train()

    while True:
        query = input("Ask me something about your JIRA issues or type 'exit': ")
        if query.lower() == "exit":
            break
        
        result = answer_question(*query.strip().split(":"))
        print(f"Predicted Answer:\n{result}\n")