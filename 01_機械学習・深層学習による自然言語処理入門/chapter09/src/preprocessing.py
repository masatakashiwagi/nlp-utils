import re

import tensorflow as tf
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
t = Tokenizer(wakati=True)


def build_vocabulary(texts, num_words=None):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, oov_token='<UNK>'
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def clean_html(html, strip=False):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text(strip=strip)
    return text


def tokenize(text):
    return t.tokenize(text)


def normalize_number(text, reduce=False):
    if reduce:
        normalized_text = re.sub(r'\d+', '0', text)
    else:
        normalized_text = re.sub(r'\d', '0', text)
    return normalized_text


def preprocess_dataset(texts):
    texts = [clean_html(text) for text in texts]
    texts = [' '.join(tokenize(text)) for text in texts]
    return texts
