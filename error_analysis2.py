import jsonlines
import datasets
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, pipeline

dataset = datasets.load_dataset("imdb")

# Load your locally saved model and tokenizer
model_path = "/scratch/general/vast/u1253335/cs6966/assignment1/models/deberta-v3-base-finetuned-imdb/checkpoint-6250"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# list of your 10 instances in the format of a dictionary {'review': <review text>, 'label': <gold label>, 'predicted': <predicted label>}
incorrect_predictions = []

trainer = Trainer(model)

# Set up a text classification pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Make predictions
# predictions = trainer.predict(encoded_dataset["unsupervised"])
predictions = classifier(encoded_dataset["unsupervised"]["text"])
print("\nprinting predictions")
print(predictions)

# Get the predicted labels
predicted_labels = [p["label"] for p in predictions]

# Get the true labels (None for unsupervised data)
true_labels = encoded_dataset["unsupervised"]["label"]

# Find indices of incorrect predictions
incorrect_indices = [
    i
    for i, (true, predicted) in enumerate(zip(true_labels, predicted_labels))
    if true is not None and true != predicted
]

# Randomly select 10 incorrect predictions
selected_incorrect_indices = random.sample(incorrect_indices, min(10, len(incorrect_indices)))

# Prepare the data for writing to the file
output_items = []

print("printing items")
for index in selected_incorrect_indices:
    item = {
        "review": encoded_dataset["unsupervised"]["text"][index],
        "label": true_labels[index],
        "predicted": predicted_labels[index],
    }
    print(item)
    output_items.append(item)

output_filename = "errors.txt"  # Give a name for the output file

# Save the selected incorrect predictions to a JSONL file
with jsonlines.open(output_filename, mode="w") as writer:
    for item in output_items:
        writer.write(item)
