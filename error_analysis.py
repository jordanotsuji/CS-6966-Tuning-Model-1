import jsonlines
import datasets
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# from huggingface_hub import login
# login()

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
predictions = trainer.predict(encoded_dataset["unsupervised"])

print(predictions.predictions[0])
print(predictions.predictions[1])
print(predictions.label_ids)
print(predictions)

# output_filename = "errors.txt"  # give a name
# # Save the selected incorrect predictions to a JSONL file
# with jsonlines.open(output_filename, mode="w") as writer:
#     for item in selected_incorrect_predictions:
#         writer.write(item)
