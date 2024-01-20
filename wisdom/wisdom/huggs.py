import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Configure logging
logging.basicConfig(level=logging.INFO)

def fine_tune_model(train_file, output_dir, model_name="gpt2", epochs=3):
    # Load pre-trained model and tokenizer
    logging.info("Loading pre-trained model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load your training data
    logging.info("Loading training data...")
    train_data = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128  # Adjust the block size based on your data size
    )

    # Create a data collator for language modeling
    logging.info("Creating data collator for language modeling...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're not using masked language modeling
    )

    # Configure training arguments
    logging.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=4,  # Adjust based on available CPU memory
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize Trainer
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
    )

    # Fine-tune the model
    logging.info("Fine-tuning the model...")
    trainer.train()

    # Save the fine-tuned model
    logging.info("Saving the fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logging.info("Fine-tuning complete.")

# Example usage:
train_file_path = 'path/to/your/training_data.txt'
output_model_dir = 'path/to/your/fine_tuned_model'

fine_tune_model(train_file_path, output_model_dir)
