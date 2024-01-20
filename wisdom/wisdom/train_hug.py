import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def fine_tune_model(train_file, output_dir, model_name="gpt2", epochs=3):
    # Load pre-trained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load your training data
    train_data = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128  # Adjust the block size based on your data size
    )

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Adjust based on available GPU memory
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Example usage:
train_file_path = 'path/to/your/training_data.txt'
output_model_dir = 'path/to/your/fine_tuned_model'

fine_tune_model(train_file_path, output_model_dir)


# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# def load_fine_tuned_model(model_path, tokenizer_path):
#     model = GPT2LMHeadModel.from_pretrained(model_path)
#     tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
#     return model, tokenizer

# def generate_answers(model, tokenizer, input_text, max_length=100, num_responses=1):
#     input_ids = tokenizer.encode(input_text, return_tensors="pt")
#     outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=num_responses)
#     generated_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
#     return generated_responses

# # Example usage:
# model_path = 'path/to/your/fine_tuned_model'
# tokenizer_path = 'path/to/your/fine_tuned_model'
# input_text = "What is the meaning of life?"

# # Load fine-tuned model and tokenizer
# fine_tuned_model, fine_tuned_tokenizer = load_fine_tuned_model(model_path, tokenizer_path)

# # Generate responses
# generated_responses = generate_answers(fine_tuned_model, fine_tuned_tokenizer, input_text, max_length=100, num_responses=1)

# # Print the generated responses
# for i, response in enumerate(generated_responses, 1):
#     print(f"Generated Response {i}: {response}")
