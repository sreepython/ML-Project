import pandas as pd

df = pd.read_csv('jira_data.csv')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if not t.lower() in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['preprocessed'] = df['description'].apply(preprocess)

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df['preprocessed'], df['label'], test_size=0.2, random_state=42)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

# Set up the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    save_total_limit=2,
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_strategy="steps",
    save_on_each_node=True,
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=X_train,
    eval_dataset=X_val,
    compute_metrics=lambda pred: {"f1": f1_score(y_val, pred["label"], average="weighted")},
)
trainer.train()

result = trainer.evaluate()
print(result)

def predict(sentence):
    # Tokenize the input sentence
    encoded_input = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    
    # Make predictions on the encoded input
    outputs = model(encoded_input["input_ids"])
    logits = outputs[0].logits
    probabilities = torch.softmax(logits, dim=-1).tolist()
    prediction = np.argmax(probabilities)
    print(prediction)