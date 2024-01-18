from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def generate_text(model_name, text, max_length=50):
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Tokenize input text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Move the model and input tensors to CPU
    model.to("cpu")
    input_ids = input_ids.to("cpu")

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            length_penalty=0.6,
            top_k=50,
            top_p=0.92,
            temperature=0.75,
            use_cache=True
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
model_name = "microsoft/phi-2"
input_text = "Once upon a time"
generated_text = generate_text(model_name, input_text)
print(generated_text)
