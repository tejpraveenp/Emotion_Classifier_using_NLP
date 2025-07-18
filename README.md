# Emotion Classifier using NLP

This project is a text classification system that identifies **emotions expressed in short sentences** using Natural Language Processing. It implements and compares three different model architectures:

- **FCNN** – Feedforward Neural Network with TF-IDF features  
- **RNN** – LSTM with Embedding Layer  
- **BERT** – Fine-tuned Transformer from HuggingFace

The models are trained, evaluated, and compared on a custom-labeled dataset containing emotional text samples.


## Utils:

- `preprocess.py` : Utility code for preprocessing data for each model before training.
- `fcnn.py`: Trainer code for FCNN model.
- `rnn.py`: Trainer code for RNN with LSTM.
- `transformer.py`: Trainer code for Transormer based BERT model.

## Notebooks:

- `01_train.ipynb` : Notebook to train all models.
- `02_Inference_and_Evaluation` : Notebook to check the inference on validation data and comparision on all models.


## Evaluation Techniques

- Classification report (accuracy, precision, recall, F1-score)

- Per-class F1 score comparison using bar plots

- Confidence score distribution analysis

- Misclassification inspection from validation predictions


---

Developed as part of an NLP project focused on emotion classification using deep learning and transformer-based models.
