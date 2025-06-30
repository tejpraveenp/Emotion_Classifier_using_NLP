import pandas as pd

def load_data(file_path):
    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ';' in line:
                text, label = line.strip().split(';')
                texts.append(text.lower())
                labels.append(label.strip().lower())

    return pd.DataFrame({'text': texts, 'label': labels})

def get_label_encoder(labels):
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label
