import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sn
import os
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


random.seed(0)
np.random.seed(0)

class_names = ['neutral', 'surprise', 'happy', 'angry', 'sad', 'fear']

num_classes = len(class_names)

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

y_test = np.array(test_data.emotion)
X_test = np.array(test_data.drop(['emotion'], axis = 1))

y_test = to_categorical(lb.fit_transform(y_test))
X_test_cnn = np.expand_dims(X_test, axis=2)

print(X_test_cnn.shape, y_test.shape)


def get_steps_number(num, bath_size):
    return num // bath_size


steps_per_epoch = get_steps_number(X_train.shape[0], train_bath_size)
validation_steps = get_steps_number(X_valid.shape[0], validation_bath_size)

print("steps per epoch: ", steps_per_epoch)
print("validation steps: ", validation_steps)


model = Sequential()

model.add(Conv1D(16, 3, activation = 'relu', padding = 'same', input_shape = (X_train_cnn.shape[1], 1)))
model.add(Conv1D(16, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling1D(2, strides = 2))

model.add(Conv1D(32, 3, activation = 'relu', padding = 'same'))
model.add(Conv1D(32, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling1D(2, strides = 2))

model.add(Conv1D(64, 3, activation = 'relu', padding = 'same'))
model.add(Conv1D(64, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling1D(2, strides = 2))

model.add(Conv1D(128, 3, activation = 'relu', padding = 'same'))
model.add(Conv1D(128, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling1D(2, strides = 2))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation = 'softmax'))

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

# evaluate the model
evaluated_model = model.evaluate(X_test)

print(evaluated_model)

# save model and weight
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = 'models'

model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# confusion matrix and classification report
Y_pred = model.predict(X_test_cnn)
y_pred = np.argmax(Y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

confusion_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred, target_names=class_names)

print(confusion_matrix)
print(classification_report)

# plt confusion matrix
df_cm = pd.DataFrame(confusion_matrix,
                     index = class_names,
                     columns = class_names)

sn.set(font_scale=1.4)
sn.heatmap(df_cm,
           annot=True,
           annot_kws={"size": 16},
           cmap="RdPu")

plt.show()