"""
Inference API.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class InferenceAPI:
    """A model API that generates output sequence.

    Attributes:
        model: Model.
        source_vocab: source language's vocabulary.
        target_vocab: target language's vocabulary.
    """

    def __init__(self, model, source_vocab, target_vocab):
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def predict_from_sequences(self, sequences):
        lengths = map(len, sequences)
        sequences = self.source_vocab.encode(sequences)
        sequences = pad_sequences(sequences, padding='post')
        y_pred = self.model.predict(sequences)
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = self.target_vocab.decode(y_pred)
        y_pred = [y[:l] for y, l in zip(y_pred, lengths)]
        return y_pred
