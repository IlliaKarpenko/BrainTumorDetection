import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator

plt.style.use("ggplot")

import cv2
from tqdm import tqdm_notebook, tnrange
from glob import glob
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, \
    MaxPooling2D, concatenate
from tensorflow.keras.optimizers import AdamW

from tensorflow.keras import backend as K

from tensorflow.python.keras import backend as keras_backend
from unet import *
from utils import *

input_height = 256
input_width = 256

training_images = []

mask_files = glob.glob('dataset/lgg-mri-segmentation/kaggle_3m/*/*_mask*')

for i in mask_files:
    training_images.append(i.replace('_mask', ''))

df = pd.DataFrame(data={'training_images': training_images, 'mask_images': mask_files})

df_train, df_test = train_test_split(df, test_size=0.1)
df_train, df_val = train_test_split(df_train, test_size=0.2)

print(df_train.shape)
print(df_test.shape)
print(df_val.shape)


def train_generator(data_frame,
                    batch_size,
                    augmentation_dict,
                    image_color_mode="rgb",
                    mask_color_mode="grayscale",
                    image_save_prefix="image",
                    mask_save_prefix="mask",
                    save_to_dir=None,
                    target_size=(256, 256),
                    seed=1, ):
    image_datagen = ImageDataGenerator(**augmentation_dict)
    mask_datagen = ImageDataGenerator(**augmentation_dict)
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col="training_images",
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed, )

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col="mask_images",
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed, )

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = normalize_and_diagnose(img, mask)
        yield img, mask


EPOCHS = 150
BATCH_SIZE = 32
initial_learning_rate = 1e-4
end_learning_rate = 1e-6
power = 1.0

smooth = 100


# Learning rate schedule function
def lr_schedule(epoch):
    return ((initial_learning_rate - end_learning_rate) *
            (1 - epoch / EPOCHS) ** power + end_learning_rate)


# Function to normalize images and masks
def normalize_and_diagnose(img, mask):
    img = img / 255.0
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return img, mask


train_generator_param = dict(rotation_range=0.2,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.05,
                             zoom_range=0.05,
                             horizontal_flip=True,
                             fill_mode='nearest')

train_gen = train_generator(df_train, BATCH_SIZE, train_generator_param, target_size=(input_height, input_width))
test_gen = train_generator(df_val, BATCH_SIZE, dict(), target_size=(input_height, input_width))

model = unet(input_size=(input_height, input_width, 3))
lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)
optimizer = AdamW(learning_rate=initial_learning_rate, weight_decay=1e-2)
model.compile(optimizer=optimizer, loss=dice_coefficients_loss,
              metrics=['binary_accuracy', intersection_over_union, dice_coefficients])

callbacks = [ModelCheckpoint('unet_model.keras', verbose=1, save_best_only=True), lr_scheduler]

# Train the model
history = model.fit(train_gen, steps_per_epoch=len(df_train) // BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks,
                    validation_data=test_gen, validation_steps=len(df_test) // BATCH_SIZE)