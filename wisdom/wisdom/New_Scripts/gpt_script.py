# pip install torch transformers pandas

import torch
from transformers import BertForQuestionAnswering, BertTokenizer
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', force_download=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)

# Function to preprocess text
def preprocess_text(text):
    return text.lower()  # Adjust preprocessing as needed

# Function to get answers based on user input
def get_answer(question, context):
    logging.debug(f"Question: {question}")
    logging.debug(f"Context: {context}")

    # Tokenize input
    inputs = tokenizer(question, context, return_tensors='pt')

    # Get model output
    outputs = model(**inputs)

    # Parse model output to get answer
    answer_start = torch.argmax(outputs['start_logits'])
    answer_end = torch.argmax(outputs['end_logits']) + 1

    answer = tokenizer.decode(inputs['input_ids'][0, answer_start:answer_end])

    logging.debug(f"Answer: {answer}")
    return answer

if __name__ == "__main__":
    # Allow the user to upload a CSV file
    csv_path = input("Please provide the path to your Jira data CSV file: ")

    # Load Jira data from CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        exit()

    # Extract relevant columns or concatenate values
    relevant_columns = ['summary', 'description', 'comments']  # Add or modify as needed
    jira_data = ' '.join([preprocess_text(str(df[column])) for column in relevant_columns])

    # Fine-tune the model (if needed) and save it
    # ...

    # Save the model and tokenizer
    model_save_path = "path_to_save_model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print(f"Model saved at {model_save_path}")

    # Allow the user to ask questions interactively
    while True:
        user_question = input("Ask a question about the Jira data (type 'exit' to end): ")

        if user_question.lower() == 'exit':
            break

        # Preprocess context based on your specific requirements
        context = ' '.join([preprocess_text(jira_data)])
        answer = get_answer(user_question, context)
        print(f"Q: {user_question}\nA: {answer}\n")
