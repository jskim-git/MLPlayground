import os
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split

from data import project_path

model_path = os.path.join(project_path, 'models')


def prepare_model(X, y, MAX_NB_WORDS, EMBEDDING_DIM):
    if os.path.isfile(os.path.join(model_path, 'abd.h5')):
        print("Found model file..")
        model = load_model(os.path.join(model_path, 'abd.h5'))
    else:
        print("No model file found.")
        print("Initializing model.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

        print("Begin training model.")
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        EPOCHS = 25
        BATCH_SIZE = 1024

        earlystop = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.001)

        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
                            callbacks=[earlystop])

        print("Saving model...")
        model.save(os.path.join(model_path, 'abd.h5'))

        acc = model.evaluate(X_test, y_test)

        print('Loss: {:0.3f}, Accuracy: {:0.3f}'.format(acc[0], acc[1]))

        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        plt.title('Accuracy')
        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.legend()
        plt.show()

    return model
