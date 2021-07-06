import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional, SpatialDropout1D

import pandas as pd
import numpy as np

import pickle




def modell(X, Y):
    num_words = 5000
    vector_space = 64
    model = Sequential()
    model.add(Embedding(num_words, vector_space, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    hist = model.fit(X, Y, epochs=20, batch_size=4, verbose=1)
    model.save('ai_model.h5', hist)


def forbered_data():
    df = pd.read_excel(r'Docs\Data.xlsx')
    varden_x = df['X'].values
    varden_y = df['Y'].values

    max_input_shape = 300
    tokenizer = Tokenizer(num_words=None, oov_token=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ", char_level=False, document_count=0)
    tokenizer.fit_on_texts(varden_x)
    ord_index = tokenizer.word_index
    X = tokenizer.texts_to_sequences(varden_x)
    X = pad_sequences(X, maxlen=max_input_shape, padding="pre", truncating="pre")
    Y = pd.get_dummies(varden_y)

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return X, Y


def main():
    data = forbered_data()
    X = data[0]
    Y = data[1]
    modell(X, Y)


if __name__ == "__main__":
    main()
