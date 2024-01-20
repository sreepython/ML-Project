import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load model and tokenizer
model_path = "model.safetensors"
config_path = "config.json"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config_path)

# Example questions to test
questions = [
    "What are the most common issues in Jira tickets?",
    "How can I prioritize critical Jira issues?",
    "What are the steps to resolve a Jira ticket?",
    "Who is responsible for resolving a specific Jira ticket?",
    "What is the status of a particular Jira issue?",
]

while True:
    user_question = input("Ask a question about your Jira data (or type 'quit' to exit): ")
    if user_question.lower() == "quit":
        break

    # Preprocess user question
    input_text = tokenizer(user_question, return_tensors="pt")

    # Get answers from the model
    with torch.no_grad():
        outputs = model(**input_text)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

    # Extract best answer span
    answer_start = torch.argmax(answer_start_scores, dim=1).item()
    answer_end = torch.argmax(answer_end_scores, dim=1).item() + 1  # Add 1 for inclusive end index
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_text["input_ids"][0][answer_start:answer_end]))

    # Print the answer
    print("Answer:", answer)
