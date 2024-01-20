import logging
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_phi2_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def fine_tune_model(train_file, output_dir, model_name="microsoft/phi-2", epochs=3):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_data = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128  # Adjust the block size based on your data size
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Adjust based on available resources
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Function to generate a response from the model
def generate_response(model, tokenizer, input_text, max_length=100):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)[0]
    generated_response = tokenizer.decode(output, skip_special_tokens=True)
    return generated_response

# Paths to CSV data and output directory for fine-tuned model
train_csv_path = 'path/to/your/data.csv'
output_model_path = 'path/to/your/fine_tuned_phi2_model'

# Fine-tune the model (train it) if not already done
# Comment out the next line if the model is already fine-tuned
fine_tune_model(train_csv_path, output_model_path, epochs=3)

# Load the fine-tuned model
phi2_model, phi2_tokenizer = load_phi2_model(output_model_path)

# Interactive testing
while True:
    user_input = input("Ask a question (or type 'exit' to stop): ")
    
    if user_input.lower() == 'exit':
        break
    
    # Use the fine-tuned model to generate a response
    generated_response = generate_response(phi2_model, phi2_tokenizer, user_input, max_length=100)
    
    # Print the generated response
    print("Generated Response:", generated_response)
