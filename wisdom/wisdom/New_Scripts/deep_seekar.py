import os
import logging as log
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

# Set up the logger
log.basicConfig(level=log.INFO)

class JiraDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len: int):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            text = str(self.data.iloc[idx])
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt')
            return {
                 'input_ids': encoding['input_ids'].flatten(),
                 'attention_mask': encoding['attention_mask'].flatten()
             }
        except Exception as e:
            log.error(f"Error occurred while processing line {idx}: {e}")

def train():
    try:
        # Load Jira Data to a pandas dataframe
        df = pd.read_csv('your_jira_data.csv', header=None, names=['text'])
        
        # Split data into training and validation sets
        train_df, val_df = train_test_split(df)
        
        model = AutoModelForSequenceClassification.from_pretrained('distsilbert-base-uncased')
        tokenizer = DistilBertTokenizerFast.from_pretrained('distsilbert-base-uncased')

        training_args = TrainingArguments(output_dir='./results', num_train_epochs=3, 
                                          per_device_train_batch_size=16, logging_steps=500, logging_dir='./logs')
        
        trainer = Trainer(model=model, args=training_args, 
                          train_dataset=JiraDataset(train_df, tokenizer), eval_dataset=JiraDataset(val_df, tokenizer))
        
        # Run the training on CPU if CUDA is not available. Uncomment below line to enable this:
        #torch.cuda.is_available() = False 

        trainer.train()
    except Exception as e:
        log.error(f"Training Failed with error {e}")

if __name__ == "__main__":
    train()