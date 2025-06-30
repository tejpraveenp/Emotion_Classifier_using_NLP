from utils.preprocess import load_data, get_label_encoder

# Load datasets
train_df = load_data("data/train.txt")
test_df = load_data("data/test.txt")

# Label encoding
label2id, id2label = get_label_encoder(train_df['label'])
train_df['label_id'] = train_df['label'].map(label2id)
test_df['label_id'] = test_df['label'].map(label2id)

print(train_df.head())
print("Labels:", label2id)
