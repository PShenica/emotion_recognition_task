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
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

# fix random seed
random.seed(0)
np.random.seed(0)

train_images_path = "dataset/spectrogram_png/train"
test_images_path = "dataset/spectrogram_png/validation"
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num_classes = len(class_names)

image_size = (64, 64)

train_bath_size = 32
test_bath_size = 14

epochs = 20

train_generator = image.ImageDataGenerator().flow_from_directory(
    train_images_path,
    classes=class_names,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=train_bath_size,
    # class_mode='categorical',
    # shuffle=False
    )

test_generator = image.ImageDataGenerator().flow_from_directory(
    test_images_path,
    classes=class_names,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=test_bath_size,
    # class_mode='categorical',
    shuffle=False
    )


def count_files_in_dir(path):
    """return number of files in directory and subdirectories"""
    number_of_files = 0

    for r, d, files in os.walk(path):
        number_of_files += len(files)

    return number_of_files


steps_per_epoch = count_files_in_dir(train_images_path) // train_bath_size
validation_steps = count_files_in_dir(test_images_path) // test_bath_size

print("steps per epoch: ", steps_per_epoch)
print("validation steps", validation_steps)

# building model
model = Sequential()

model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (image_size[0], image_size[1], 1)))
model.add(Conv2D(16, (3, 3), activation = 'relu', padding = 'same', name = 'block0_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block0_pool1'))

model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv1'))
model.add(Conv2D(32, (3, 3), activation = 'relu', padding = 'same', name = 'block1_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block1_pool1'))


model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv1'))
model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same', name = 'block2_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block2_pool1'))

model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv1'))
model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same', name = 'block3_conv2'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), name = 'block3_pool1'))


model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4, name = 'Dropout_5'))
model.add(Dense(num_classes, activation = 'softmax'))

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
test_generator.reset()
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

# confusion matrix and classification report
test_generator.reset()
Y_pred = model.predict_generator(test_generator, validation_steps + 1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
confusion_matrix = confusion_matrix(test_generator.classes, y_pred)
print(confusion_matrix)

print('Classification Report')
classification_report = classification_report(test_generator.classes, y_pred, target_names=class_names)
print(classification_report)

# plt confusion matrix
df_cm = pd.DataFrame(confusion_matrix,
                     index = class_names,
                     columns = class_names)

sn.set(font_scale=1.4)  # for label size
sn.heatmap(df_cm,
           annot=True,
           annot_kws={"size": 16},  # font size
           cmap="RdPu")

plt.show()




