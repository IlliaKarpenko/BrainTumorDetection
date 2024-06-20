import cv2
import os
from PIL import Image
import numpy as np

images_dir = 'dataset/'

no_tumor_images = os.listdir(images_dir + 'no/')
yes_tumor_images = os.listdir(images_dir + 'yes/')
dataset = []
label = []

for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(images_dir + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(images_dir + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)