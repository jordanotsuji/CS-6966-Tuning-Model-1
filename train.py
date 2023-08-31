import transformers
import datasets
import argparse
import os

from huggingface_hub import login

# notebook_login()
login(token="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")

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="Directory where model checkpoints will be saved")
args = parser.parse_args()

# GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "imdb"
model_checkpoint = "microsoft/deberta-v3-base"
batch_size = 12

from datasets import load_dataset, load_metric

# actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("plain_text", task)
dataset = load_dataset(task)
# metric = load_metric("imdb", actual_task)

import numpy as np

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
}

sentence1_key, sentence2_key = task_to_keys[task]


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)


encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)


model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    os.path.join(output_dir, f"{model_name}-finetuned-{task}"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)


validation_key = (
    "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
)
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
