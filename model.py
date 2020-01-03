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

# fix random seed
random.seed(0)
np.random.seed(0)

num_classes = 7
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

image_size = (64, 64)
batch_size = 32
steps_per_epoch = 40  # train image count / bath size = steps per epoch
validation_steps = 8  # val image count / bath size = validation steps
epochs = 5

train_generator = image.ImageDataGenerator().flow_from_directory(
    "dataset/spectrogram_png/train",
    classes=class_names,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size)

test_generator = image.ImageDataGenerator().flow_from_directory(
    "dataset/spectrogram_png/validation",
    classes=class_names,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size)

# building model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# compile model
opt = keras.optimizers.Adam(learning_rate = 0.0003)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
cnn_history = model.fit_generator(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps)

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
