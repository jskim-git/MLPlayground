import pandas as pd
import numpy as np
import itertools

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from cnn_ra import MyConv as cnn

sns.set(style='white', palette='deep')

seed = 2020

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

y_train = df_train["label"]
X_train = df_train.drop(labels=["label"], axis=1)

# How does original data look like?
# X_t = X_train.values.reshape(-1, 28, 28, 1)
# plt.imshow(X_t[0][:, :, 0])
# plt.show()

# Free memory space (optional)
del df_train

# Normalization [0, 255] => [0, 1]
# Why? Faster convergence

X_train = X_train / 255
df_test = df_test / 255

"""
Reshaping the image dataset.
The train dataset contains individual pixel values from 0 - 783
Instead, we try rebuilding the dataset to a grayscale image.
(Vector of length 784 => 3D matrix 28*28*1, image length of 28 by 28)

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
 |   |   |   |       |   | 
756 757 758 759 ... 782 783

Wait, why 28*28*1? What's the 1? Represents the channel.
MNIST dataset here is gray scaled, hence only one channel here.
For RGB colored images, there are 3 channels (R, G, B) => 28*28*3
"""

X_train = X_train.values.reshape(-1, 28, 28, 1)
df_test = df_test.values.reshape(-1, 28, 28, 1)

print(X_train.shape)

# So how does the data look like?
plt.imshow(X_train[2][:, :, 0])
plt.show()

# From 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] etc. (One-hot encoding?)
y_train = to_categorical(y_train, num_classes=10)  # Digits are from 0 to 9

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

"""
Convolutional Neural Networks (CNNs) using Keras Sequential API

First layer: Convolutional layer (Conv2D)
- Basically, a set of learnable FILTERS. These filters transform a part of the image by the kernel size, which is 
  applied to the whole image (basically a transformation). This allows the CNN to isolate features that are useful
  into a feature map!

Second layer: Pooling layer (MaxPool2D)
- This is where we downsample the filter. Amongst the pixels, it selects the maximum value. This allows us to reduce
  computation times, while also reduce overfitting (by underfitting? wut). Here, we simply choose the pooling size,
  which is the pooled area size.
  
With layer 1 + layer 2, CNNs can combine useful local features into a feature map, then learn about the features of 
the whole image!

Dropout
- Dropout is a regularization method. A sample of nodes in the layer are "randomly" ignored, setting their weights to
  zero. Doing this improves the generalization powers of our CNN model.
  
ReLU
- Activation function, where f(x) = max(0, x).

Flatten
- Flatten the feature map into a single 1D vector. This is required, as it combines all the found local features of
  our previous convolutional layers.
"""

# Begin CNN model construction here!
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=X_train.shape[1:4]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())

# Final layers, just a simple ANN classifier!
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

# Optimizer and scorers
# RMSProp is an updater that adjusts the AdaGrad method in simple ways. SGD (Stochastic Gradient Descent) also usable
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Set up a function to define a decreasing learning rate
lr_reduct = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=1e-05)
EPOCHS = 1
BATCH_SIZE = 64

cnn = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                validation_data=(X_val, y_val), verbose=1,
                callbacks=[lr_reduct])

# Plotting the loss / accuracy curves for training and validation
fig, ax = plt.subplots(2, 1)
ax[0].plot(cnn.history['loss'], color='b', label="Training loss")
ax[0].plot(cnn.history['val_loss'], color='r', label="Validation loss", axes=ax[0])
legend = ax[0].legend(loc='best', shadow=True)
plt.show()

ax[1].plot(cnn.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(cnn.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()


# Confusion matrix (Define a function that prints / plots the confusion matrix)

def conf_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.tight_layout()
    plt.ylabel('True Category')
    plt.xlabel('Predicted Category')
    plt.show()


y_pred = model.predict(X_val)
y_bin = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_bin)
conf_matrix(cm, classes=range(10))

# What causes these errors?
error = (y_bin - y_true != 0)
y_bin_error = y_bin[error]
y_pred_error = y_pred[error]
y_true_error = y_true[error]
X_val_error = X_val[error]


# Define a function to show 6 images that were falsely classified
def show_errors(error_index, error_img, error_pred, error_obs):
    n = 0
    nrow = 2
    ncol = 3
    fig, ax = plt.subplots(nrow, ncol, sharex=True, sharey=True)
    for row in range(nrow):
        for col in range(ncol):
            error = error_index[n]
            ax[row, col].imshow((error_img[error]).reshape((28, 28)))
            ax[row, col].set_title("Predicted: {}\nTrue: {}".format(error_pred[error], error_obs[error]))
            n += 1
    plt.show()


y_pred_error_prob = np.max(y_pred_error, axis=1)
true_prob_error = np.diagonal(np.take(y_pred_error, y_true_error, axis=1))
delta_pred_true_error = y_pred_error_prob - true_prob_error
sorted_delta_error = np.argsort(delta_pred_true_error)

most_important_errors = sorted_delta_error[-6:]
show_errors(most_important_errors, X_val_error, y_bin_error, y_true_error)
