from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertJapaneseTokenizer

from models import build_model, loss_func
from preprocessing import convert_examples_to_features, Vocab, preprocess_dataset
from utils import load_dataset, evaluate


def main():
    # Set hyper-parameters.
    batch_size = 32
    epochs = 100
    model_path = 'models/'
    pretrained_model_name_or_path = 'bert-base-japanese-whole-word-masking'
    maxlen = 250

    # Data loading.
    x, y = load_dataset('./data/ja.wikipedia.conll')
    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path, do_word_tokenize=False)

    # Pre-processing.
    x = preprocess_dataset(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    target_vocab = Vocab(lower=False).fit(y_train)
    features_train, labels_train = convert_examples_to_features(
        x_train,
        y_train,
        target_vocab,
        max_seq_length=maxlen,
        tokenizer=tokenizer
    )
    features_test, labels_test = convert_examples_to_features(
        x_test,
        y_test,
        target_vocab,
        max_seq_length=maxlen,
        tokenizer=tokenizer
    )

    # Build model.
    model = build_model(pretrained_model_name_or_path, target_vocab.size)
    model.compile(optimizer='sgd', loss=loss_func(target_vocab.size))

    # Preparing callbacks.
    callbacks = [
        EarlyStopping(patience=3),
    ]

    # Train the model.
    model.fit(x=features_train,
              y=labels_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.1,
              callbacks=callbacks,
              shuffle=True)
    model.save_pretrained(model_path)

    # Evaluation.
    evaluate(model, target_vocab, features_test, labels_test)


if __name__ == '__main__':
    main()
