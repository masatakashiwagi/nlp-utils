"""
Preprocessings.
"""
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table


def build_vocabulary(text, num_words=None):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<UNK>')
    tokenizer.fit_on_texts([text])
    return tokenizer


def create_dataset(text, vocab, num_words, window_size, negative_samples):
    data = vocab.texts_to_sequences([text]).pop()
    sampling_table = make_sampling_table(num_words)
    couples, labels = skipgrams(data, num_words,
                                window_size=window_size,
                                negative_samples=negative_samples,
                                sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.reshape(word_target, (-1, 1))
    word_context = np.reshape(word_context, (-1, 1))
    labels = np.asarray(labels)
    return [word_target, word_context], labels
