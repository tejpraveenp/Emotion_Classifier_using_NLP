import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def train_rnn_model(train_df, test_df, label2id, maxlen=50, vocab_size=10000, embed_dim=100, epochs=10):
    # Tokenize text
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df['text'])

    X_train_seq = tokenizer.texts_to_sequences(train_df['text'])
    X_test_seq = tokenizer.texts_to_sequences(test_df['text'])

    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

    y_train = to_categorical(train_df['label_id'], num_classes=len(label2id))
    y_test = to_categorical(test_df['label_id'], num_classes=len(label2id))

    # Build RNN model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(label2id), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train on full training data
    model.fit(X_train_pad, y_train, epochs=epochs, batch_size=32, verbose=2)

    # Evaluate on test data
    y_pred = model.predict(X_test_pad)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = test_df['label_id'].values

    print("\nLSTM RNN Classification Report:\n")
    print(classification_report(y_true, y_pred_labels, target_names=label2id.keys(), zero_division=0))

    return model
