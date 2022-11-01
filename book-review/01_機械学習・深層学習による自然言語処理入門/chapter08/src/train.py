from pprint import pprint

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from inference import InferenceAPI
from model import EmbeddingModel
from preprocessing import build_vocabulary, create_dataset
from utils import load_data


def main():
    # Set hyper-parameters.
    emb_dim = 50
    epochs = 10
    model_path = 'model.h5'
    negative_samples = 1
    num_words = 10000
    window_size = 1

    # Load a dataset.
    text = load_data(filepath='data/ja.text8')

    # Build a vocabulary.
    vocab = build_vocabulary(text, num_words)

    # Create a training dataset.
    x, y = create_dataset(text, vocab, num_words, window_size, negative_samples)

    # Build a simple model.
    model = EmbeddingModel(num_words, emb_dim)
    model = model.build()
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # Prepare callbacks.
    callbacks = [
        EarlyStopping(patience=1),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    # Train the model.
    model.fit(x=x,
              y=y,
              batch_size=128,
              epochs=epochs,
              validation_split=0.2,
              callbacks=callbacks)

    # Inference.
    model = load_model(model_path)
    api = InferenceAPI(model, vocab)
    pprint(api.most_similar(word='日本'))


if __name__ == '__main__':
    main()
