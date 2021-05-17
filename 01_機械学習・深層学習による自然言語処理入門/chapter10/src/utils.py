"""
Utilities.
"""
import numpy as np
from seqeval.metrics import classification_report


def load_dataset(filename, encoding='utf-8'):
    """Loads data and label from a file.
    Args:
        filename (str): path to the file.
        encoding (str): file encoding format.
        The file format is tab-separated values.
        A blank line is required at the end of a sentence.
        For example:
        ```
        EU	B-ORG
        rejects	O
        German	B-MISC
        call	O
        to	O
        boycott	O
        British	B-MISC
        lamb	O
        .	O
        Peter	B-PER
        Blackburn	I-PER
        ...
        ```
    Returns:
        tuple(numpy array, numpy array): data and labels.
    Example:
        >>> filename = 'conll2003/en/ner/train.txt'
        >>> data, labels = load_data_and_labels(filename)
    """
    sents, labels = [], []
    words, tags = [], []
    with open(filename, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if line:
                word, tag = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                sents.append(words)
                labels.append(tags)
                words, tags = [], []
        if words:
            sents.append(words)
            labels.append(tags)

    return sents, labels


def evaluate(model, target_vocab, features, labels):
    label_ids = model.predict(features)
    label_ids = np.argmax(label_ids, axis=-1)
    y_pred = [[] for _ in range(label_ids.shape[0])]
    y_true = [[] for _ in range(label_ids.shape[0])]
    for i in range(label_ids.shape[0]):
        for j in range(label_ids.shape[1]):
            if labels[i][j] == 0:
                continue
            y_pred[i].append(label_ids[i][j])
            y_true[i].append(labels[i][j])
    y_pred = target_vocab.decode(y_pred)
    y_true = target_vocab.decode(y_true)
    print(classification_report(y_true, y_pred, digits=4))
