import re

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Vocab:

    def __init__(self, num_words=None, lower=True, oov_token=None):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=num_words,
            oov_token=oov_token,
            filters='',
            lower=lower,
            split='\t'
        )

    def fit(self, sequences):
        texts = self._texts(sequences)
        self.tokenizer.fit_on_texts(texts)
        return self

    def encode(self, sequences):
        texts = self._texts(sequences)
        return self.tokenizer.texts_to_sequences(texts)

    def decode(self, sequences):
        texts = self.tokenizer.sequences_to_texts(sequences)
        return [text.split(' ') for text in texts]

    def _texts(self, sequences):
        return ['\t'.join(words) for words in sequences]

    def get_index(self, word):
        return self.tokenizer.word_index.get(word)

    @property
    def size(self):
        """Return vocabulary size."""
        return len(self.tokenizer.word_index) + 1

    def save(self, file_path):
        with open(file_path, 'w') as f:
            config = self.tokenizer.to_json()
            f.write(config)

    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(f.read())
            vocab = cls()
            vocab.tokenizer = tokenizer
        return vocab


def normalize_number(text, reduce=True):
    if reduce:
        normalized_text = re.sub(r'\d+', '0', text)
    else:
        normalized_text = re.sub(r'\d', '0', text)
    return normalized_text


def preprocess_dataset(sequences):
    sequences = [[normalize_number(w) for w in words] for words in sequences]
    return sequences


def create_dataset(sequences, vocab):
    sequences = vocab.encode(sequences)
    sequences = pad_sequences(sequences, padding='post')
    return sequences


def convert_examples_to_features(x, y,
                                 vocab,
                                 max_seq_length,
                                 tokenizer):
    pad_token = 0
    features = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'label_ids': []
    }
    for words, labels in zip(x, y):
        tokens = [tokenizer.cls_token]
        label_ids = [pad_token]
        for word, label in zip(words, labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            label_id = vocab.get_index(label)
            label_ids.extend([label_id] + [pad_token] * (len(word_tokens) - 1))
        tokens += [tokenizer.sep_token]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        token_type_ids = [pad_token] * max_seq_length

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)
        features['label_ids'].append(label_ids)

    for name in features:
        features[name] = pad_sequences(features[name], padding='post', maxlen=max_seq_length)

    x = [features['input_ids'], features['attention_mask'], features['token_type_ids']]
    y = features['label_ids']
    return x, y
