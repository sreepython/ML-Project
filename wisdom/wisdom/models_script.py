import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_name, user_input):
    # Set device for the model
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Set device for the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Encode user input
    inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=False)

    # Move inputs to the device
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    # Generate text
    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

    # Split the text based on empty lines and select the first non-empty part
    answers = [part.strip() for part in text.split('\n\n') if part.strip()]
    
    # Extract the first answer if available
    first_answer = answers[0] if answers else "No answer found."
    
    return first_answer

# Example usage
# model_name = "microsoft/phi-2"
# input_text = "What are you?"
# first_answer = generate_text(model_name, input_text)
# print(first_answer)
