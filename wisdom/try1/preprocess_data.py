import logging
from transformers import BertTokenizer
import pandas as pd
import re


def clean_and_sanitize_text(text, stop_words):
    """
    Cleans and sanitizes text by lowercasing, removing punctuation, and optionally removing stop words.

    Args:
        text (str): The text to be cleaned and sanitized.
        stop_words (list[str]): A list of stop words to remove.

    Returns:
        str: The cleaned and sanitized text.
    """
    logging.debug("Cleaning and sanitizing text: %s", text)
    print(text)
    if pd.isna(text):
        text = "Missing Description"
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    if stop_words:
        text = " ".join([word for word in text.split() if word not in stop_words])
    logging.debug("Cleaned and sanitized text: %s", text)
    return text


def load_and_clean_data(filepath, remove_stop_words=False):
    logging.info("Loading data from CSV file: %s", filepath)
    data = pd.read_csv(filepath)
    cleaned_questions = []
    cleaned_answers = []
    stop_words = ["a", "an", "the", "...", "etc."]
    for index, row in data.iterrows():
        if pd.isna(row["Description"]):
            logging.debug("Skipping row %d: 'Description' contains 'nan'", index + 1)
            continue
        logging.debug("Processing row %d", index + 1)
        question = clean_and_sanitize_text(row["Summary"], stop_words)
        answer = clean_and_sanitize_text(row["Description"], stop_words)
        cleaned_questions.append(question)
        cleaned_answers.append(answer)
    logging.info("Data cleaning and sanitization complete! Returned cleaned data.")
    return {"cleaned_questions": cleaned_questions, "cleaned_answers": cleaned_answers}


def tokenize_and_encode_data(cleaned_questions, cleaned_answers):
    """
    Tokenizes and encodes a list of questions and answers using a pretrained BERT tokenizer.

    Args:
        cleaned_questions (list[str]): A list of cleaned and sanitized questions.
        cleaned_answers (list[str]): A list of cleaned and sanitized answers.

    Returns:
        dict: A dictionary containing two lists: encoded_questions and encoded_answers.
    """
    logging.info("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    encoded_questions = []
    encoded_answers = []
    for question, answer in zip(cleaned_questions, cleaned_answers):
        logging.debug("Tokenizing question: %s", question)
        question_encoded = tokenizer.encode(question, add_special_tokens=True)
        logging.debug("Encoded question: %s", question_encoded)
        logging.debug("Tokenizing answer: %s", answer)
        answer_encoded = tokenizer.encode(answer, add_special_tokens=True)
        logging.debug("Encoded answer: %s", answer_encoded)
        encoded_questions.append(question_encoded)
        encoded_answers.append(answer_encoded)
    logging.info("Tokenization and encoding complete! Returning encoded data.")
    return {"encoded_questions": encoded_questions, "encoded_answers": encoded_answers}


# Example usage
logging.basicConfig(level=logging.INFO)
cleaned_data = load_and_clean_data(r"C:\Users\sree\ML-Project\wisdom\try1\jira_data.csv", remove_stop_words=True)
encoded_data = tokenize_and_encode_data(cleaned_data["cleaned_questions"], cleaned_data["cleaned_answers"])

print(cleaned_data)
print("=================================================================")
print(encoded_data)
