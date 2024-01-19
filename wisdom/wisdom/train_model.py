from transformers import LlamaTokenizer, LlamaForCausalLM
import torch



# Download and load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# Read the file
with open('C:\Users\sree\ML-Project\wisdom\wisdom\train_text_ram.txt', 'r') as f:
    text = f.read()
    
# Tokenize inputs
inputs = tokenizer(text, return_tensors="pt", max_length=512) # replace 'yourfile.txt' with your file path

# Forward pass
outputs = model(**inputs)