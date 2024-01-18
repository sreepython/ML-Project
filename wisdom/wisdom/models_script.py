import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model_name, user_input):
    torch.set_default_device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)

    # Use a regular expression to identify and extract answers
    answers = re.split(r'\d+\. ', text)
    
    # Extract the first answer if available
    first_answer = answers[1].strip() if len(answers) > 1 else "No answer found."
    
    return first_answer

# # Example usage
# model_name = "microsoft/phi-2"
# input_text = "What is convergence in networking?"
# first_answer = generate_text(model_name, input_text)
# print(first_answer)
