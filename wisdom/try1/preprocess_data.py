import logging
from transformers import BertTokenizer
import pandas as pd
import re


def preprocess_bert_data(filepath, remove_stopwords=False):
    """
    Preprocesses text data from a CSV file for BERT model with detailed logging.

    Args:
        filepath: Path to the CSV file.
        remove_stopwords: Boolean flag to indicate whether to remove stop words.

    Returns:
        A dictionary containing two lists:
            questions: List of preprocessed and encoded questions.
            answers: List of preprocessed and encoded answers.
    """

    logger = logging.getLogger(__name__)
    logger.info("Starting BERT data preprocessing for file: %s", filepath)

    # Load data from CSV
    logger.info("Loading data from CSV...")
    data = pd.read_csv(filepath)

    # Initialize BERT tokenizer
    logger.info("Initializing BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Initialize empty lists for preprocessed data
    logger.info("Initializing empty lists for preprocessed data...")
    questions = []
    answers = []

    # Loop through each row in the data
    for index, row in data.iterrows():
        # Get question and answer from specific columns
        logger.info("Processing row %d...", index + 1)
        question = row["Summary"]
        answer = row.get("Description", "")  # Use get with default to avoid attribute error

        # Skip rows with empty answers
        if not answer:
            logger.debug("Skipping row %d: empty answer found", index + 1)
            continue

        # Check and handle non-string answers
        if not isinstance(answer, str):
            logger.debug("Skipping row %d: answer is not a string", index + 1)
            continue

        # Lowercase and remove punctuation
        logger.debug("Lowercasing and removing punctuation...")
        question = question.lower()
        answer = answer.lower()
        question = re.sub(r"^[\W\s]+$", "", question)
        answer = re.sub(r"^[\W\s]+$", "", answer)

        # Optionally remove stop words
        if remove_stopwords:
            logger.debug("Removing stop words (if enabled)...")
            stop_words = ["a", "an", "the", "...", "etc."]
            question = " ".join([word for word in question.split() if word not in stop_words])
            answer = " ".join([word for word in answer.split() if word not in stop_words])

        # Tokenize question and answer
        logger.debug("Tokenizing question and answer...")
        question_encoded = tokenizer.encode(question, add_special_tokens=True)
        answer_encoded = tokenizer.encode(answer, add_special_tokens=True)

        # Append to respective lists
        logger.debug("Appending preprocessed data to lists...")
        questions.append(question_encoded)
        answers.append(answer_encoded)

    # Return preprocessed data as a dictionary
    logger.info("Preprocessing complete! Returning data...")
    return {"questions": questions, "answers": answers}


# Example usage
logging.basicConfig(level=logging.INFO)
preprocessed_data = preprocess_bert_data(r"C:\Users\sree\ML-Project\wisdom\try1\jira_data.csv", remove_stopwords=True)

print(preprocessed_data)