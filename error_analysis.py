import jsonlines
import datasets
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# from huggingface_hub import login
# login()

dataset = datasets.load_dataset("imdb")

# Load your locally saved model and tokenizer
model_path = "/scratch/general/vast/u1253335/cs6966/assignment1/models/deberta-v3-base-finetuned-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# list of your 10 instances in the format of a dictionary {'review': <review text>, 'label': <gold label>, 'predicted': <predicted label>}
incorrect_predictions = []


# Make predictions and evaluate
for sample in dataset["unsupervised"]:
    text = sample["text"]  # Assuming the "unsupervised" portion contains text data
    # Tokenize and encode the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Make predictions
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = logits.argmax().item()

    # Since this is the "unsupervised" portion, there are no true labels.
    # We consider predictions with high confidence (e.g., confidence < -1.0) as incorrect.
    if logits.max().item() > -1.0:
        incorrect_predictions.append({"review": text, "label": sample["label"], "predicted": predicted_label})

# Randomly select 10 incorrect predictions
random.seed(42)  # Set seed for reproducibility
selected_incorrect_predictions = random.sample(incorrect_predictions, min(10, len(incorrect_predictions)))


output_filename = "errors.txt"  # give a name
# Save the selected incorrect predictions to a JSONL file
with jsonlines.open(output_filename, mode="w") as writer:
    for item in selected_incorrect_predictions:
        writer.write(item)
