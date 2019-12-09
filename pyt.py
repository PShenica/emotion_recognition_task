import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# fix random seed
random.seed(0)
np.random.seed(0)

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

train = image.ImageDataGenerator().flow_from_directory("Dataset/spectrogram/train", classes=class_names, target_size=(34, 50), color_mode='grayscale', batch_size=4)
test = image.ImageDataGenerator().flow_from_directory("Dataset/spectrogram/val", classes=class_names, target_size=(34, 50), color_mode='grayscale', batch_size=4)

"""
    # dividing the data into test and train
    train = pd.read_csv('Dataset/train.csv')
    val = pd.read_csv('Dataset/validation.csv')
    
    train_features = train.iloc[:, 1:-1]
    train_label = train.iloc[:, -1:]
    
    test_features = val.iloc[:, 1:-1]
    test_label = val.iloc[:, -1:]
    
    X_train = np.array(train_features)
    y_train = np.array(train_label)
    X_test = np.array(test_features)
    y_test = np.array(test_label)
    
    # categorize data
    lb = LabelEncoder()
    
    Y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    
    
    # padding sequence for CNN model
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
"""

# building model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(34, 50, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# compile model
opt = keras.optimizers.adam()
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
cnn_history = model.fit_generator(train, validation_data=test, epochs=10)
# cnn_history = model.fit(X_train, y_train, batch_size=10, epochs=100, validation_data=(X_test, y_test), verbose=1)

# save model and weight
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'models'

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# show graphs of accuracy and loss
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
