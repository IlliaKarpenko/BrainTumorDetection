import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
plt.style = ("ggplot")


def plot_images_with_masks(num_rows, num_columns, image_paths, mask_paths):
    figure = plt.figure(figsize=(12, 12))

    for i in range(1, num_rows * num_columns + 1):
        figure.add_subplot(num_rows, num_columns, i)

        image_path = image_paths[i]
        mask_path = mask_paths[i]

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)

        plt.imshow(image_rgb)
        plt.imshow(mask, alpha=0.4)

    plt.show()

def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten + K.sum(y_pred_flatten))
    return (2*intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def intersection_over_union(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -intersection_over_union(y_true_flatten, y_pred_flatten)