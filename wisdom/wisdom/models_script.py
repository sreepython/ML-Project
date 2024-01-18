import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



def generate_text(model_name, user_input):
    torch.set_default_device("cpu")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    inputs = tokenizer(user_input, return_tensors="pt", return_attention_mask=False)

    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    print(text)
    return text
