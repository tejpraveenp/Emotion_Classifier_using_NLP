import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

def train_fcnn_model(train_df, test_df, label2id, max_features=5000, epochs=10):
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=max_features)
    X_train = tfidf.fit_transform(train_df['text']).toarray()
    X_test = tfidf.transform(test_df['text']).toarray()

    y_train = to_categorical(train_df['label_id'], num_classes=len(label2id))
    y_test = to_categorical(test_df['label_id'], num_classes=len(label2id))

    # FCNN model
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(len(label2id), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train on full training data (no validation)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=2)

    # Evaluate on test data
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true = test_df['label_id'].values

    print("\nFCNN Classification Report:\n")
    print(classification_report(y_true, y_pred_labels, target_names=label2id.keys(), zero_division=0))

    return model
