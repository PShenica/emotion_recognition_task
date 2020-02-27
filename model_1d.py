import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sn
import os
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# np.set_printoptions(threshold = np.inf)

# fix random seed
random.seed(0)
np.random.seed(0)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

num_classes = len(class_names)
image_size = (64, 64)

train_bath_size = 32
validation_bath_size = 14

epochs = 20


data = pd.read_csv("dataset/csv/data.csv")

data = data.sample(frac = 1)
test_data = data[3000:]
data = data[:3000]

y = data.emotion
X = data.drop(['emotion'], axis = 1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

print(X_valid.shape, X_train.shape)

lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_valid = to_categorical(lb.fit_transform(y_valid))

X_train_cnn = np.expand_dims(X_train, axis=2)
X_valid_cnn = np.expand_dims(X_valid, axis=2)


def get_steps_number(num, bath_size):
    return num // bath_size


steps_per_epoch = get_steps_number(X_train.shape[0], train_bath_size)
validation_steps = get_steps_number(X_valid.shape[0], validation_bath_size)

print("steps per epoch: ", steps_per_epoch)
print("validation steps: ", validation_steps)


model = Sequential()
model.add(Conv1D(256, 8, padding='same', input_shape = (X_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(7))
model.add(Activation('softmax'))

model.summary()

opt = keras.optimizers.Adam(learning_rate = 0.001)
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.compile(loss = 'categorical_crossentropy',
              optimizer = opt,
              metrics = ["accuracy"])

cnn_history = model.fit(
    X_train_cnn,
    y_train,
    validation_data = (X_valid_cnn, y_valid),
    epochs = epochs,
    steps_per_epoch = steps_per_epoch,
    validation_steps = validation_steps,
    callbacks=[es_callback])


plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

