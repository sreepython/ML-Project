import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(level=logging.INFO)

def load_pretrained_model():
    try:
        model = GPT2LMHeadModel.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        tokenizer = GPT2Tokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None, None

def read_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = [line.strip() for line in file.readlines()]
        return data
    except Exception as e:
        logging.error(f"Error reading data from file: {str(e)}")
        return None

def generate_responses(model, tokenizer, data):
    try:
        responses = []
        for input_text in data:
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)
            generated_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            responses.append(generated_response)
        return responses
    except Exception as e:
        logging.error(f"Error generating responses: {str(e)}")
        return None

def main():
    model, tokenizer = load_pretrained_model()

    if model is None or tokenizer is None:
        logging.error("Exiting due to model loading error.")
        return

    data = read_data('path/to/your/data.txt')

    if data is None:
        logging.error("Exiting due to data reading error.")
        return

    responses = generate_responses(model, tokenizer, data)

    if responses is None:
        logging.error("Exiting due to response generation error.")
        return

    for i, (input_text, response) in enumerate(zip(data, responses), 1):
        logging.info(f"Input {i}: {input_text}")
        logging.info(f"Generated Response {i}: {response}")
        logging.info("=" * 50)

if __name__ == "__main__":
    main()
