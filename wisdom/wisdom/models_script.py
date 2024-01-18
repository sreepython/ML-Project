from transformers import AutoTokenizer, AutoModel
import torch

def generate_text(model_name, text, max_length=50, vocab_size=50308):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Check if GPU is available
    if torch.cuda.is_available():
        print("Using GPU for computation...")
        device = torch.device('cuda')
        model.to(device)
    else:
        print("GPU not available, falling back to CPU...")
        device = torch.device('cpu')

    # Tokenize input text
    input_ids = tokenizer.encode(text + ' ', return_tensors="pt").to(device)

    # Initialize empty list for storing tokens
    sequence = []

    # Generate text using the model
    with torch.no_grad():
        hidden_states = model(input_ids)[0]

    # Process hidden states
    for idx in range(hidden_states.shape[-1]):
        logits = hidden_states[:, -1, :]  # Adjusted line

        # Ensure logits are not nan or inf
        if torch.isnan(logits).any() or torch.isinf(logits).any() or (logits < 0).any():
            print("Invalid logits detected. Breaking the loop.")
            break

        next_token_logits = logits[:, :]  # Corrected line

        # Ensure next_token_logits are not nan or inf
        if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any() or (next_token_logits < 0).any():
            print("Invalid next_token_logits detected. Breaking the loop.")
            break

        # Sample the next token index
        sampled_index = torch.multinomial(next_token_logits, num_samples=1, replacement=True).item()

        # Get the corresponding word
        word = tokenizer.decode(sampled_index)

        # Append the word to the sequence
        sequence.append(word)

        # Stop processing once the end token is found
        if word == " ":
            break

    # Format the result
    generated_text = "\n".join(sequence[:-1])
    assistant_prompt = f"\nAssistant:\n{generated_text}"
    user_prompt = f"\nUser: {text}\n"
    combined_response = user_prompt + assistant_prompt
    return combined_response

# Example usage
# model_name = "microsoft/phi-2"
# input_text = "What is convergence in networking?"
# generated_text = generate_text(model_name, input_text)
# print(generated_text)

You: What is convergence in networking?
Server: User: What is convergence in networking? Assistant:
You: what are you?
Server: User: what are you? Assistant: 

This what I am getting fine tune it and get full answer 