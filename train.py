import transformers
import datasets
import argparse
import os
import jsonlines

from huggingface_hub import login

# notebook_login()
login(token="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")


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, help="Directory where model checkpoints will be saved")
args = parser.parse_args()

# GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

task = "imdb"
model_checkpoint = "microsoft/deberta-v3-base"
batch_size = 4

from datasets import load_dataset, load_metric

# actual_task = "mnli" if task == "mnli-mm" else task
# dataset = load_dataset("plain_text", task)
dataset = load_dataset(task)
metric = load_metric("accuracy")

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
    os.path.join(args.output_dir, f"{model_name}-finetuned-{task}"),
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


# validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
validation_key = "test" if task == "imdb" else "validation"
trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
predictions, labels, metrics = trainer.predict(encoded_dataset["unsupervised"])
print("\n\nMetrics: ----------------------------------------------\n")
print(metrics)
# print first 10 predictions
print("\n\nPredictions: ----------------------------------------------\n")
print(predictions[:10])

filename = "errors.txt"  # give a name
# list of your 10 instances in the format of a dictionary {'review': <review text>, 'label': <gold label>, 'predicted': <predicted label>}
output_items = []
bad_predictions = []

for prediction in predictions:
    # append if prediction label wasn't the same as real label
    if prediction.label != prediction.predictions:
        bad_predictions.append(prediction)

# randomly add 10 bad predictions to output_items
for i in range(10):
    # random number of size bad_predictions.size
    j = np.random.randint(0, len(bad_predictions) - 1)
    output_items.append(
        {
            "review": bad_predictions[j].sentence,
            "label": bad_predictions[j].label,
            "predicted": bad_predictions[j].predictions,
        }
    )


with jsonlines.open(filename, mode="w") as writer:
    for item in output_items:
        writer.write(item)

# I = (
#     {'train': Dataset(features: {'sentence': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None), 'idx': Value(dtype='int32', id=None)}, num_rows: 8551),
#      'validation': Dataset(features: {'sentence': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None), 'idx': Value(dtype='int32', id=None)}, num_rows: 1043),
#      'test': Dataset(features: {'sentence': Value(dtype='string', id=None), 'label': ClassLabel(num_classes=2, names=['unacceptable', 'acceptable'], names_file=None, id=None), 'idx': Value(dtype='int32', id=None)}, num_rows: 1063)})

# Some weights of DebertaV2ForSequenceClassification were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['classifier.weight', 'pooler.dense.weight', 'classifier.bias', 'pooler.dense.bias']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
#   0%|          | 0/6250 [00:00<?, ?it/s]You're using a DebertaV2TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
# 100%|█████████▉| 6249/6250 [18:00<00:00,  6.17it/s]{'loss': 0.3875, 'learning_rate': 1.8400000000000003e-05, 'epoch': 0.08}
# {'loss': 0.3171, 'learning_rate': 1.6800000000000002e-05, 'epoch': 0.16}
# {'loss': 0.3057, 'learning_rate': 1.5200000000000002e-05, 'epoch': 0.24}
# {'loss': 0.26, 'learning_rate': 1.3600000000000002e-05, 'epoch': 0.32}
# {'loss': 0.2679, 'learning_rate': 1.2e-05, 'epoch': 0.4}
# {'loss': 0.232, 'learning_rate': 1.04e-05, 'epoch': 0.48}
# {'loss': 0.2411, 'learning_rate': 8.8e-06, 'epoch': 0.56}
# {'loss': 0.209, 'learning_rate': 7.2000000000000005e-06, 'epoch': 0.64}
# {'loss': 0.2163, 'learning_rate': 5.600000000000001e-06, 'epoch': 0.72}
# {'loss': 0.2001, 'learning_rate': 4.000000000000001e-06, 'epoch': 0.8}
# {'loss': 0.2023, 'learning_rate': 2.4000000000000003e-06, 'epoch': 0.88}
# {'loss': 0.1843, 'learning_rate': 8.000000000000001e-07, 'epoch': 0.96}
# 100%|██████████| 6250/6250 [23:29<00:00,  4.43it/s]
# {'eval_loss': 0.1816762536764145, 'eval_accuracy': 0.96188, 'eval_runtime': 322.8182, 'eval_samples_per_second': 77.443, 'eval_steps_per_second': 19.361, 'epoch': 1.0}
# {'train_runtime': 1409.9686, 'train_samples_per_second': 17.731, 'train_steps_per_second': 4.433, 'train_loss': 0.2497784002685547, 'epoch': 1.0}
# 100%|██████████| 6250/6250 [05:22<00:00, 19.39it/s]
# ^C
