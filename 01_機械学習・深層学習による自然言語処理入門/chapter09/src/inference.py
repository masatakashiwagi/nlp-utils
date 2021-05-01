"""
Inference API.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class InferenceAPI:
    """A model API that generates output sequence.

    Attributes:
        model: Model.
        vocab: language's vocabulary.
    """

    def __init__(self, model, vocab, preprocess):
        self.model = model
        self.vocab = vocab
        self.preprocess = preprocess

    def predict_from_texts(self, texts):
        x = self.preprocess(texts)
        x = self.vocab.texts_to_sequences(x)
        return self.predict_from_sequences(x)

    def predict_from_sequences(self, sequences):
        sequences = pad_sequences(sequences, truncating='post')
        y = self.model.predict(sequences)
        return np.argmax(y, -1)
