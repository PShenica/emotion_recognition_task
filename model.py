import numpy as np
import random
import matplotlib.pyplot as plt
import os
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.metrics import classification_report

# fix random seed
random.seed(0)
np.random.seed(0)

num_classes = 7
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

image_size = (64, 64)

train_bath_size = 32
test_bath_size = 14

steps_per_epoch = 40  # train image count / train bath size = steps per epoch
validation_steps = 19  # val image count / test bath size = validation steps

epochs = 15

train_generator = image.ImageDataGenerator().flow_from_directory(
    "dataset/spectrogram_png/train",
    classes=class_names,
    target_size=image_size,
    # color_mode='grayscale',
    batch_size=train_bath_size)

test_generator = image.ImageDataGenerator().flow_from_directory(
    "dataset/spectrogram_png/validation",
    classes=class_names,
    target_size=image_size,
    # color_mode='grayscale',
    batch_size=test_bath_size)

# building model
model = Sequential()

model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (image_size[0], image_size[1], 3)))
model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'block0_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block0_pool1'))
model.add(Dropout(0.2, name = 'Dropout_1'))

model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1'))
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool1'))


model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1'))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool1'))
model.add(Dropout(0.4, name = 'Dropout_3'))

model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool1'))


model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4, name = 'Dropout_5'))
model.add(Dense(num_classes, activation = 'sigmoid'))

model.summary()

# compile model
opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
cnn_history = model.fit_generator(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2)

# evaluate the model
evaluated_model = model.evaluate_generator(generator=test_generator, steps=validation_steps)
print(evaluated_model)

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

