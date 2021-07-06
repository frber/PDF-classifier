import os
import pdfplumber
import openpyxl
from nltk.corpus import stopwords
import re
import shutil

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional, SpatialDropout1D
from keras.utils import to_categorical

from tensorflow.keras.models import load_model

import pandas as pd
import numpy as np

import pickle





def predict(rensad_text, text, tabell, model, tokenizer, filvag):
    text_l = []
    text_l.append(rensad_text)
    lista_text = []
    lista_text.append(text.lower())
    seq = tokenizer.texts_to_sequences(text_l)
    pad_text = pad_sequences(seq, maxlen=300)
    pred = model.predict(pad_text)
    labels = ['Interreg', 'Vinnova', 'VR']
    finansiar = labels[np.argmax(pred)]
    procent = np.max(pred)
    print(finansiar, procent)

def rensa_text(text):
    text = ''.join([i for i in text if not i.isdigit()]) # Ta bort siffror
    stoppord = set(stopwords.words('swedish'))
    text = ' '.join(word for word in text.split() if word not in stoppord) # Ta bort stoppord
    return text


def hamta_text(filvag):
    lista_text = []
    lista_tabell = []
    with pdfplumber.open(filvag) as pdf:
        sidor = pdf.pages
        for sida in sidor:
            text = sida.extract_text()
            if text != None:
                lista_text.append(text)

            tabell = sida.extract_table()
            if tabell != None:
                lista_tabell.append(tabell)

    fulltext = ' '.join(lista_text)
    return fulltext, lista_tabell


def testa_oppna(filvag):
    try:
        open(filvag, 'r')
    except:
        return True


def sok_pdf(filvag, model, tokenizer):
    for root, dirs, files in os.walk(filvag):
        for fil in files:
            fil = fil.lower()
            if fil.endswith(".pdf"):
                filvag = os.path.join(root, fil)
                if testa_oppna(filvag):
                    print("FEL", filvag)
                    continue
                else:
                    hamta = hamta_text(filvag)
                    text = hamta[0]
                    tabell = hamta[1]
                    rensad_text = rensa_text(text)
                    predict(rensad_text, text, tabell, model, tokenizer, filvag)


def ladda_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def ladda_modell():
    model = load_model('ai_model.h5')
    return model


def main():
    filvag = r'C:\Users\berfre\Desktop\testpdf'
    model = ladda_modell()
    tokenizer = ladda_tokenizer()
    sok_pdf(filvag, model, tokenizer)



if __name__ == "__main__":
    main()


