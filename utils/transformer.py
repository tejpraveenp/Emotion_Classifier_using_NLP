import os
import numpy as np
import pandas as pd
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": (preds == labels).mean()
    }

def train_transformer_model(train_df, test_df, label2id, max_length=128, epochs=3, models_dir="models"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = EmotionDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = EmotionDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id)
    )

    training_args = TrainingArguments(
        output_dir=models_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        no_cuda=False,
        save_strategy="no",  # we're saving manually after training
        logging_dir=os.path.join(models_dir, "logs"),
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Trainer is using device:", trainer.args.device)

    trainer.train()

    # Evaluate and print classification report
    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    true = test_df["label_id"].values

    print("\nBERT Classification Report:\n")
    print(classification_report(true, preds, target_names=label2id.keys(), zero_division=0))

    return model, tokenizer
