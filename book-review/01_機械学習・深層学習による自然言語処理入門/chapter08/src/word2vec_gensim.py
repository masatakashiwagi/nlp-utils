"""
Word2vec model.
"""
import argparse
import logging
from pprint import pprint

from gensim.models.word2vec import Word2Vec, Text8Corpus
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train(args):
    sentences = Text8Corpus(args.data_path)
    model = Word2Vec(sentences, size=args.dimension, window=args.window)
    model.save(args.save_file)


def calculate_similar_words(model):
    word = input('Input a word: ')
    pprint(model.most_similar(word))


def calculate_word_similarity(model):
    words_str = input('Input two words(e.g. 猫,犬): ')
    word1, word2 = words_str.split(',')
    print(model.similarity(word1, word2))


def predict(args):
    print('Loading model...')
    model = KeyedVectors.load(args.save_file)
    while True:
        print('m: Calculate similar words.')
        print('s: Calculate word similarity.')
        print('e: Exit.')
        choice = input('Your choice: ')
        if choice == 'm':
            calculate_similar_words(model)
        elif choice == 's':
            calculate_word_similarity(model)
        elif choice == 'e':
            break
        else:
            print('Please input m, s or e!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training a word2vec model')
    parser.add_argument('--data_path', default='data/ja.text8', help='path to dataset')
    parser.add_argument('--save_file', default='model/model.bin', help='save file')
    parser.add_argument('--dimension', default=100, type=int, help='embedding dimension')
    parser.add_argument('--window', default=5, type=int, help='window size')
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
