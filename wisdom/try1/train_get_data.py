import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data.dataset import ConcatDataset, TensorDataset

# Import your data preprocessing functions
from pp2nd_data import load_and_preprocess_data, tokenize_and_encode_data, split_data

def pad_labels(labels, num_classes):
    padded_labels = torch.zeros((len(labels), num_classes), dtype=torch.float32)
    for i, label in enumerate(labels):
        for class_index in label:
            padded_labels[i, class_index] = 1
    return padded_labels

def train_model(train_data, val_data, epochs=4, batch_size=16, learning_rate=2e-5, target_length=None):
    logging.info("Initializing model and tokenizer...")
    logging.info(f"train_data: {train_data}")
    
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Calculate the maximum length of labels
    max_len_of_labels = max(len(label) for label in train_data["labels"])
    
    train_data["labels"] = pad_labels(train_data["labels"], target_length=max_len_of_labels)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    tensor_dataset = TensorDataset(
        torch.tensor(train_data["input_ids"], dtype=torch.long),
        torch.tensor(train_data["labels"], dtype=torch.long)
    )
    
    replication_factor = 15
    replicated_train_data = ConcatDataset([tensor_dataset] * replication_factor)
    train_dataloader = DataLoader(replicated_train_data, batch_size=batch_size, shuffle=True)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    if len(train_dataloader.dataset) == 0:
        logging.error("Training dataset is empty. Check your data loading and preprocessing.")
        return
    else:
        logging.info(f"Training dataset size: {len(train_dataloader.dataset)}")

    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        for batch in train_dataloader:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            logits = outputs.last_hidden_state
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

        logging.info(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        # Validation phase
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_predictions = 0

            for batch in val_dataloader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                outputs = model(input_ids)
                logits = outputs.last_hidden_state
                loss = criterion(logits, labels)

                total_loss += loss.item()

                predicted = torch.argmax(logits, dim=1)
                total_correct += torch.sum((predicted == labels)).item()
                total_predictions += labels.size(0)

            avg_loss = total_loss / len(val_dataloader)
            accuracy = total_correct / total_predictions

            logging.info(f"Epoch {epoch+1} Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, f"checkpoint-epoch{epoch}.pth")

    torch.save(model.state_dict(), "trained_model.pth")

# Assuming preprocess_text is defined somewhere
def answer_user_questions(model, tokenizer, device):
    while True:
        question = input("Ask a question: ")
        processed_question = preprocess_text(question)
        input_ids = tokenizer.encode(processed_question, return_tensors="pt")
        input_ids = input_ids.to(device)

        with torch.no_grad():
            outputs = model(input_ids)

        logits = outputs.last_hidden_state
        predicted = torch.argmax(logits, dim=1)
        answer = tokenizer.decode(predicted, skip_special_tokens=True)
        print("Answer:", answer)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load and preprocess data
    preprocessed_data = load_and_preprocess_data(r"C:\Users\sree\ML-Project\wisdom\try1\jira_data.csv", remove_stop_words=True)
    encoded_data = tokenize_and_encode_data(preprocessed_data["cleaned_questions"], preprocessed_data["cleaned_answers"])
    train_data, val_data, test_data = split_data(encoded_data)

    # Calculate max_len_of_labels
    max_len_of_labels = max(len(label) for label in train_data["labels"])

    num_classes = 768
    train_model(train_data, val_data, target_length=max_len_of_labels, num_classes=num_classes)
    # Train the model
    # # train_model(train_data, val_data, target_length=max_len_of_labels)

    # Load the trained model and answer user questions
    model = BertModel.from_pretrained("bert-base-uncased")
    model.load_state_dict(torch.load("trained_model.pth"))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Assuming preprocess_text is defined somewhere
    answer_user_questions(model, tokenizer, device)
