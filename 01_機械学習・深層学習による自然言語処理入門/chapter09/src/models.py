from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D


class RNNModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       weights=[embeddings],
                                       name='embedding')
        self.rnn = SimpleRNN(hid_dim, name='rnn')
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.rnn(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class LSTMModel:

    def __init__(self, input_dim, output_dim,
                 emb_dim=300, hid_dim=100,
                 embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       trainable=trainable,
                                       weights=[embeddings],
                                       name='embedding')
        self.lstm = LSTM(hid_dim, name='lstm')
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        output = self.lstm(embedding)
        y = self.fc(output)
        return Model(inputs=x, outputs=y)


class CNNModel:

    def __init__(self, input_dim, output_dim,
                 filters=250, kernel_size=3,
                 emb_dim=300, embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       trainable=trainable,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       trainable=trainable,
                                       weights=[embeddings],
                                       name='embedding')
        self.conv = Conv1D(filters,
                           kernel_size,
                           padding='valid',
                           activation='relu',
                           strides=1)
        self.pool = GlobalMaxPooling1D()
        self.fc = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        conv = self.conv(embedding)
        pool = self.pool(conv)
        y = self.fc(pool)
        return Model(inputs=x, outputs=y)


class LSTMCNNModel:

    def __init__(self, input_dim, output_dim,
                 filters=250, kernel_size=3,
                 emb_dim=300, hid_dim=100, embeddings=None):
        self.input = Input(shape=(None,), name='input')
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       mask_zero=True,
                                       name='embedding')
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       mask_zero=True,
                                       weights=[embeddings],
                                       name='embedding')
        self.lstm = LSTM(hid_dim, return_sequences=True, name='lstm')
        self.conv = Conv1D(filters,
                           kernel_size,
                           padding='valid',
                           activation='relu',
                           strides=1)
        self.pool = GlobalMaxPooling1D()
        self.fc1 = Dense(hid_dim)
        self.fc2 = Dense(output_dim, activation='softmax')

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        lstm = self.lstm(embedding)
        conv = self.conv(lstm)
        pool = self.pool(conv)
        y = self.fc1(pool)
        y = self.fc2(y)
        return Model(inputs=x, outputs=y)
