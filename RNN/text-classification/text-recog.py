import pandas as pd
import numpy as np
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, GlobalMaxPool1D

project_path = Path(__file__).resolve().parents[2]
data_path = os.path.join(project_path, 'RNN', 'text-classification', 'data')

filepath_dict = {'yelp': os.path.join(data_path, 'yelp_labelled.txt'),
                 'amazon': os.path.join(data_path, 'amazon_cells_labelled.txt'),
                 'imdb': os.path.join(data_path, 'imdb_labelled.txt')}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source
    df_list.append(df)

df = pd.concat(df_list)
# print(df.iloc[0])

# Single model
# df_yelp = df[df['source'] == 'yelp']
# sentences = df_yelp['sentence']
# y = df_yelp['label'].values
#
# sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=2020)
#
# vectorizer = CountVectorizer()
# vectorizer.fit(sentences_train)
#
# X_train = vectorizer.transform(sentences_train)
# X_test = vectorizer.transform(sentences_test)
# print(X_train)

# model = LogisticRegression()
# model.fit(X_train, y_train)
# score = model.score(X_test, y_test)

# print("Accuracy: {:.3f}".format(score))

sources = df['source'].unique()
for source in sources:
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.2, random_state=2020)
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    # X_train = vectorizer.transform(sentences_train)
    # X_test = vectorizer.transform(sentences_test)

    # # Logistic model
    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # score = model.score(X_test, y_test)
    # print("Accuracy for {} data: {:.3f}".format(source, score))

    # # Keras model
    EPOCHS = 250
    BATCH_SIZE = 50
    # input_dim = X_train.shape[1]
    #
    # model = Sequential()
    # model.add(Dense(10, input_dim=input_dim, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))  # Sigmoid for prob.
    #
    # model.compile(loss='mse', optimizer='rmsprop', metrics=['mse'])
    # model.summary()
    #
    # history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=False, batch_size=BATCH_SIZE,
    #                     validation_data=(X_test, y_test))
    #
    # print(model.evaluate(X_train, y_train, verbose=False))

    # Keras with Word Embedding (Tokenizer)
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1

    # print(sentences_train[2])
    # print(X_train[2])

    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    # print(X_train[0, :])

    embedding_dim = 50
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(GlobalMaxPool1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=False,
                        validation_data=(X_test, y_test), batch_size=BATCH_SIZE)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print('Test Accuracy: {:.4f}'.format(accuracy))
    print(loss)


