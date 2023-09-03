import jsonlines
import datasets
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

dataset = datasets.load_dataset("imdb")

# Load your locally saved model and tokenizer
model_path = "/scratch/general/vast/u1253335/cs6966/assignment1/models/deberta-v3-base-finetuned-imdb/checkpoint-6250"
# model_path = "./checkpoint-6250"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# list of your 10 instances in the format of a dictionary {'review': <review text>, 'label': <gold label>, 'predicted': <predicted label>}
incorrect_predictions = []

trainer = Trainer(model)

# Make predictions
predictions = trainer.predict(encoded_dataset["test"])


# Get the predicted labels
predicted_labels = [p.argmax() for p in predictions.predictions]

# Get the true labels (None for unsupervised data)
true_labels = encoded_dataset["test"]["label"]

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

for index in selected_incorrect_indices:
    item = {
        "review": encoded_dataset["test"]["text"][index],
        "label": str(true_labels[index]),
        "predicted": str(predicted_labels[index]),
    }
    print(item)
    output_items.append(item)

output_filename = "errors2.jsonl"  # Give a name for the output file

# Save the selected incorrect predictions to a JSONL file
with jsonlines.open(output_filename, mode="w") as writer:
    for item in output_items:
        writer.write(item)

# print accuracy of predictions
print(
    "Accuracy: ",
    sum([1 for i, (true, predicted) in enumerate(zip(true_labels, predicted_labels)) if true == predicted])
    / len(true_labels),
)

# print number of wrong predictions
print("Number of wrong predictions: ", len(incorrect_indices))
