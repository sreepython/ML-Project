from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_fine_tuned_model(model_path, tokenizer_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer

def generate_answers(model, tokenizer, input_text, max_length=100, num_responses=1):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=num_responses)
    generated_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_responses

# Example usage:
model_path = 'path/to/your/fine_tuned_model'
tokenizer_path = 'path/to/your/fine_tuned_model'
input_text = "What is the capital of France?"

# Load fine-tuned model and tokenizer
fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_path, tokenizer_path)

# Generate responses
generated_responses = generate_answers(fine_tuned_model, fine_tuned_tokenizer, input_text, max_length=100, num_responses=1)

# Print the generated responses
for i, response in enumerate(generated_responses, 1):
    print(f"Generated Response {i}: {response}")
