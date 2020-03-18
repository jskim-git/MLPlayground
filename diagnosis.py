import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from data import load_dataset
from model import prepare_model

df = load_dataset()

# LSTM Modeling
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 115
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df["IPTN_CNCS_CNTE"].values)
word_index = tokenizer.word_index

X = tokenizer.texts_to_sequences(df["IPTN_CNCS_CNTE"].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

y = df["FATTY_DEG"]
y = pd.get_dummies(y)

model = prepare_model(X, y, MAX_NB_WORDS, EMBEDDING_DIM)


def interact():
    print("Enter input diagnosis: ")
    new = [input()]
    seq = tokenizer.texts_to_sequences(new)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
    pred = np.round(model.predict(padded), 4)
    # labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'No degree']
    labels = [0, 1, 2, 3, 9]
    index = int(np.argmax(pred))
    print(pred, labels[index])
    print('fatty_deg: {}'.format(labels[index]))


while True:
    interact()
