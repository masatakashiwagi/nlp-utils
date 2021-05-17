from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from seqeval.metrics import classification_report

from inference import InferenceAPI
from models import UnidirectionalModel, BidirectionalModel
from preprocessing import create_dataset, preprocess_dataset, Vocab
from utils import load_dataset


def main():
    # Set hyper-parameters.
    batch_size = 32
    epochs = 100
    model_path = 'models/model_{}.h5'
    num_words = 15000

    # Data loading.
    x, y = load_dataset('./data/ja.wikipedia.conll')

    # Pre-processing.
    x = preprocess_dataset(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    source_vocab = Vocab(num_words=num_words, oov_token='<UNK>').fit(x_train)
    target_vocab = Vocab(lower=False).fit(y_train)
    x_train = create_dataset(x_train, source_vocab)
    y_train = create_dataset(y_train, target_vocab)

    # Build models.
    models = [
        UnidirectionalModel(num_words, target_vocab.size).build(),
        BidirectionalModel(num_words, target_vocab.size).build(),
    ]
    for i, model in enumerate(models):
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Preparing callbacks.
        callbacks = [
            EarlyStopping(patience=3),
            ModelCheckpoint(model_path.format(i), save_best_only=True)
        ]

        # Train the model.
        model.fit(x=x_train,
                  y=y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.1,
                  callbacks=callbacks,
                  shuffle=True)

        # Inference.
        model = load_model(model_path.format(i))
        api = InferenceAPI(model, source_vocab, target_vocab)
        y_pred = api.predict_from_sequences(x_test)
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()
