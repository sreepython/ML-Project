import torch
from transformers import AutoModelForQuestionAnswering, DistilBertTokenizerFast

def train_model(model, tokenized_texts):
    """Trains the model on the provided data and generates answers."""

    optimizer = transformers.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):
        print(f"Epoch {epoch+1}:")

        # Forward pass
        outputs = model(**tokenized_texts)
        print(f"Outputs: {outputs}")  # Inspect model output

        # Check for loss
        if "loss" not in outputs:
            print("Warning: Loss not found in model output. Check model configuration and loss function.")
            continue  # Skip backward pass and optimization if loss is missing

        loss = outputs.loss
        print(f"Loss: {loss}")

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1} completed.")

        # Generate answers after each epoch (optional)
        generate_answers(model, tokenizer)  # Call the answer generation function

# Function to generate answers
def generate_answers(model, tokenizer):
    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == "quit":
            break

        # Preprocess user question
        input_text = tokenizer(user_question, return_tensors="pt")

        # Get answers from the model
        with torch.no_grad():
            outputs = model(**input_text)
            answer_start_scores = outputs.start_logits
            answer_end_scores = outputs.end_logits

        # Extract best answer span and decode
        answer_start = torch.argmax(answer_start_scores, dim=1).item()
        answer_end = torch.argmax(answer_end_scores, dim=1).item() + 1  # Add 1 for inclusive end index
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_text["input_ids"][0][answer_start:answer_end]))

        # Print the answer
        print("Answer:", answer)

# Train the model (replace placeholders with actual paths and data)
model_path = "path/to/your/model"
config_path = "path/to/your/config"
text_iterator = ...  # Replace with your training data iterator

tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
tokenizer.train_new_from_iterator(text_iterator, vocab_size=config.vocab_size)  # Adjust config.vocab_size if needed
tokenizer.save_pretrained(model_path)

model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config_path)

train_model(model, tokenized_texts)  # Replace tokenized_texts with your prepared data
