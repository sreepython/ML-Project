import logging
from transformers import BertTokenizer
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



def preprocess_text(text, stop_words):
    """
    Cleans and sanitizes text by lowercasing, removing punctuation, and optionally removing stop words.

    Args:
        text (str): The text to be cleaned and sanitized.
        stop_words (list[str]): A list of stop words to remove.

    Returns:
        str: The cleaned and sanitized text.
    """
    logging.debug("Cleaning and sanitizing text: %s", text)
    if pd.isna(text):
        text = "Missing Description"
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    if stop_words:
        text = " ".join([word for word in text.split() if word not in stop_words])
    logging.debug("Cleaned and sanitized text: %s", text)
    return text


def load_and_preprocess_data(filepath, remove_stop_words=False):
    """
    Loads data from a CSV file, cleans and sanitizes it, and prepares it for BERT.

    Args:
        filepath (str): Path to the CSV file.
        remove_stop_words (bool, optional): Whether to remove stop words. Defaults to False.

    Returns:
        dict: A dictionary containing preprocessed questions and answers.
    """
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
        question = preprocess_text(row["Summary"], stop_words)
        answer = preprocess_text(row["Description"], stop_words)
        cleaned_questions.append(question)
        cleaned_answers.append(answer)
    logging.info("Data preprocessing complete! Returned preprocessed data.")
    return {"cleaned_questions": cleaned_questions, "cleaned_answers": cleaned_answers}


def tokenize_and_encode_data(cleaned_questions, cleaned_answers):
    """
    Tokenizes and encodes a list of preprocessed questions and answers using a BERT tokenizer.

    Args:
        cleaned_questions (list[str]): A list of cleaned and sanitized questions.
        cleaned_answers (list[str]): A list of cleaned and sanitized answers.

    Returns:
        dict: A dictionary containing encoded questions and answers, without padding.
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
    logging.info("Tokenization and encoding complete! Returned encoded data.")
    return {"input_ids": encoded_questions, "labels": encoded_answers}


def split_data(preprocessed_data, test_size=0.2, val_size=0.1):
    """
    Splits preprocessed data into training, validation, and test sets.

    Args:
        preprocessed_data (dict): A dictionary containing preprocessed questions, answers, and attention masks.
        test_size (float, optional): Proportion of data for the test set. Defaults to 0.2.
        val_size (float, optional): Proportion of data for the validation set. Defaults to 0.1.

    Returns:
        tuple: A tuple containing dictionaries for training, validation, and test data.
    """
    logging.info("Splitting data into training, validation, and test sets...")

    # Combine questions and answers into a single array for splitting
    X = preprocessed_data["input_ids"]
    y = preprocessed_data["labels"]

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_remaining, y_remaining, test_size=val_size / (1 - test_size), random_state=42)

    train_data = {"input_ids": X_train, "labels": y_train}
    val_data = {"input_ids": X_val, "labels": y_val}
    test_data = {"input_ids": X_test, "labels": y_test}

    logging.info("Data split complete! Returned training, validation, and test data.")
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Example usage
    # Example usage
    logging.basicConfig(level=logging.INFO)
    preprocessed_data = load_and_preprocess_data(r"C:\Users\sree\ML-Project\wisdom\try1\jira_data.csv", remove_stop_words=True)
    max_len = 512

    # encoded_data = tokenize_and_encode_data(preprocessed_data["cleaned_questions"], preprocessed_data["cleaned_answers"], max_len=max_len, padding="max_length")
    # Replace "encoded_data" with your actual encoded data dictionaries for questions and answers
    encoded_data = tokenize_and_encode_data(preprocessed_data["cleaned_questions"], preprocessed_data["cleaned_answers"])

    # Pad sequences to the desired maximum length (adjust to your needs)
    padded_questions = pad_sequences(encoded_data["input_ids"], maxlen=max_len, padding="post", value=0)
    padded_answers = pad_sequences(encoded_data["labels"], maxlen=max_len, padding="post", value=0)

    # Update the encoded data dictionary with the padded sequences
    encoded_data["input_ids"] = padded_questions
    encoded_data["labels"] = padded_answers

    # Now you have preprocessed, encoded, and padded data ready for training!

    # Remove the following lines, as there is no 'encoded_questions' key in your dictionary
    # encoded_questions = encoded_data["encoded_questions"]
    # encoded_answers = encoded_data["encoded_answers"]

    train_data, val_data, test_data = split_data(encoded_data)

    print(preprocessed_data)
    print("=================================================================")
    print(encoded_data)
    print("=================================================================")
    print("Training data:", train_data)
    print("=================================================================")
    print("Validation data:", val_data)
    print("=================================================================")
    print("Test data:", test_data)
