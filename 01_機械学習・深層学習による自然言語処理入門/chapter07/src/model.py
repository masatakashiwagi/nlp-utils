from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def create_model(vocab_size, label_size, hidden_size=16):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape=(vocab_size,)))
    model.add(Dense(label_size, activation='softmax'))
    return model
