"""
Inference API.
"""
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


class InferenceAPI:
    """A model API that generates output sequence.

    Attributes:
        model: Model.
        vocab: vocabulary.
    """

    def __init__(self, model, vocab):
        self.vocab = vocab
        self.weights = model.get_layer('word_embedding').get_weights()[0]

    def most_similar(self, word, topn=10):
        word_index = self.vocab.word_index.get(word, 1)
        sim = self._cosine_similarity(word_index)
        pairs = [(s, i) for i, s in enumerate(sim)]
        pairs.sort(reverse=True)
        pairs = pairs[1: topn + 1]
        res = [(self.vocab.index_word[i], s) for s, i in pairs]
        return res

    def similarity(self, word1, word2):
        word_index1 = self.vocab.word_index.get(word1, 1)
        word_index2 = self.vocab.word_index.get(word2, 1)
        weight1 = self.weights[word_index1]
        weight2 = self.weights[word_index2]
        return cosine(weight1, weight2)

    def _cosine_similarity(self, target_idx):
        target_weight = self.weights[target_idx]
        similarity = cosine_similarity(self.weights, [target_weight])
        return similarity.flatten()
