import logging
import pandas as pd
import transformers
import torch

# Configure logging
logging.basicConfig(filename='jira_analysis.log', level=logging.INFO)

def load_data(csv_path):
    """Loads Jira data from CSV and handles potential errors."""
    try:
        data = pd.read_csv(csv_path)
        logging.info(f"Data loaded successfully from {csv_path}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {csv_path}")
        raise

def filter_data(data, filter_criteria):
    """Filters data based on specified criteria."""
    try:
        filtered_data = data.query(filter_criteria)
        logging.info(f"Data filtered successfully with criteria: {filter_criteria}")
        return filtered_data
    except pd.errors.EmptyDataError:
        logging.warning(f"Warning: Filter criteria resulted in empty DataFrame")
        return data  # Return original data if filtering leads to empty result

def preprocess_text(texts):
    """Preprocesses text data for model input."""
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    return tokenized_texts

def train_model(model, tokenized_texts):
    """Trains the model on the provided data."""
    optimizer = transformers.AdamW(model.parameters(), lr=5e-5)
    for epoch in range(3):
        # Forward pass
        outputs = model(**tokenized_texts)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logging.info(f"Model training completed successfully")

def save_model(model, model_path):
    """Saves the trained model to disk."""
    model.save_pretrained(model_path)
    logging.info(f"Model saved to {model_path}")

def main():
    """Main execution flow."""
    csv_path = "jira_data.csv"  # Replace with your CSV path
    filter_criteria = "priority == 'Critical' & status == 'Open'"  # Adjust as needed
    model_name = 'distilbert-base-uncased'
    model_path = "jira_trained_model"

    try:
        data = load_data(csv_path)
        filtered_data = filter_data(data, filter_criteria)

        # Extract relevant text fields based on Jira details
        text_fields = ["Summary", "Description", "Comment", "Environment", "Steps to Reproduce", "Resolution"]
        additional_fields = ["Reporter", "Assignee", "Labels", "Issue Type", "Priority", "Status"]

        # Combine selected text fields
        texts = filtered_data[text_fields].to_numpy().flatten().tolist()  # Combine primary text fields
        texts.extend(filtered_data[additional_fields].to_numpy().flatten().tolist())  # Optionally add additional fields

        tokenized_texts = preprocess_text(texts)
        model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
        train_model(model, tokenized_texts)
        save_model(model, model_path)

        logging.info("Script execution completed successfully")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
