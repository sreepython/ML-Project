import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import logging
import re

logging.basicConfig(filename='training.log', level=logging.INFO)

def load_data(csv_path):
    """Loads the CSV data and logs its structure."""
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV loaded with columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        logging.error("CSV file not found. Please check the path.")
        return None

def preprocess_data(df):
    """Preprocesses the data, handling dynamic columns and data types."""
    relevant_columns = ["summary", "description", "status", "priority", "assignee", "comments"]  # Adjust as needed

    # Check for missing columns:
    missing_columns = [col for col in relevant_columns if col not in df.columns]
    if missing_columns:
        logging.warning(f"Missing columns: {missing_columns}")

    # Keep only relevant columns present in the CSV:
    df = df[relevant_columns.intersection(df.columns)]

    # Preprocess text data:
    df.text = df.text.str.lower()  # Lowercase
    df.text = df.text.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove punctuation

    # Handle varying data types:
    # ... (Adapt preprocessing for different data types)

    return df

def create_qa_pairs(df):
    """Manually creates question-answer pairs based on example use cases."""
    qa_pairs = []
    for i, row in df.iterrows():
        qa_pairs.append({
            "question": "What are the critical issues?",
            "answer": row["summary"] if row["priority"] == "Critical" else "None"
        })
        # ... Add more pairs for other use cases ...
    return qa_pairs

def train_model(model, tokenizer, qa_pairs):
    """Fine-tunes the model and saves it."""
    try:
        # Tokenize text and create attention masks
        train_encodings = tokenizer(qa_pairs["question"], qa_pairs["answer"], truncation=True, padding=True)

        # Fine-tune the model
        model.fit(train_encodings["input_ids"], train_encodings["attention_mask"], epochs=3)  # Adjust epochs as needed

        # Save the model
        model.save_pretrained("my_jira_model")

        logging.info("Model training completed and saved successfully.")
    except Exception as e:
        logging.error("Error during model training:", e)

def answer_question(question, context, model, tokenizer):
    """Predicts and returns the answer to a user question based on the model and context."""
    # Tokenize question and context
    inputs = tokenizer(question, context, return_tensors="pt")

    # Pass to model and get predicted answer
    outputs = model(**inputs)
    answer_start_scores, answer_end_scores = outputs.start_logits, outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    return answer

# ... Remaining code (user input prompt and answer display) ...
if __name__ == "__main__":
    csv_path = "jira_data.csv"

    # ... (Load and process data, create QA pairs, train and save model) ...

    # Load the saved model and tokenizer
    model = AutoModelForQuestionAnswering.from_pretrained("my_jira_model")
    tokenizer = AutoTokenizer.from_pretrained("my_jira_model")

    # Prompt the user for questions
    while True:
        question = input("Enter your question about Jira data: ")

        # Use the "summary" column as context for answering questions
        context = df["summary"].tolist()  # Adjust as needed

        try:
            answer = answer_question(question, context, model, tokenizer)
            print("Answer:", answer)
        except Exception as e:
            logging.error("Error answering question:", e)
            print("An error occurred while answering the question. Please try again.")
