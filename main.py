import cv2
import os

from keras import Input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.src.layers import Conv2D, Activation, Dropout, Flatten, Dense
from keras.src.layers import MaxPooling2D
from keras.src.utils import normalize, to_categorical
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential


images_dir = 'dataset/'

no_tumor_images = os.listdir(images_dir + 'no/')
yes_tumor_images = os.listdir(images_dir + 'yes/')
dataset = []
label = []

INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(images_dir + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(images_dir + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# building a model
model = Sequential()

model.add(Input(shape=(INPUT_SIZE, INPUT_SIZE, 3)))

model.add(Conv2D(int(INPUT_SIZE / 2), (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(INPUT_SIZE, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(INPUT_SIZE))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)
model.save('model_dump.keras')
