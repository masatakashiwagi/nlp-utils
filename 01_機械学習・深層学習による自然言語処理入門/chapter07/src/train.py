from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from model import create_model
from preprocessing import clean_html, tokenize
from utils import load_dataset, plot_history


def main():
    # Loading dataset.
    x, y = load_dataset('data/amazon_reviews_multilingual_JP_v1_00.tsv', n=5000)

    # Preprocessing dataset.
    x = [clean_html(text, strip=True) for text in x]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Vectorizing dataset.
    vectorizer = CountVectorizer(tokenizer=tokenize)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train = x_train.toarray()
    x_test = x_test.toarray()

    # Setting hyperparameters
    vocab_size = len(vectorizer.vocabulary_)
    label_size = len(set(y_train))

    # Building a model
    model = create_model(vocab_size, label_size)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    filepath = 'model.h5'
    callbacks = [
        EarlyStopping(patience=3),
        ModelCheckpoint(filepath, save_best_only=True),
        TensorBoard(log_dir='logs')
    ]

    # Training a model
    history = model.fit(x_train, y_train,
                        validation_split=0.2,
                        epochs=100,
                        batch_size=32,
                        callbacks=callbacks)

    # train_seq = Generator(x_train, y_train)
    # valid_seq = Generator(x_test, y_test)
    # history = model.fit_generator(generator=train_seq,
    #                               validation_data=valid_seq,
    #                               epochs=100,
    #                               callbacks=callbacks,
    #                               shuffle=True)

    model = load_model(filepath)

    text = 'このアプリ超最高！'
    vec = vectorizer.transform([text])
    y_pred = model.predict(vec.toarray())
    print(y_pred)

    acc = model.evaluate(x_test, y_test)
    print(acc)

    plot_history(history)


if __name__ == '__main__':
    main()
