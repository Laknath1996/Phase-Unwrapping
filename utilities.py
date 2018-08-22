# author : Ashwin de Silva
# some helper functions for the main modules

# import libraries

import cv2
import numpy as np
import tensorflow as tf
import h5py


# define the helper functions

def load_image(addr, size):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr, 0)
    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_h5py_datasets(PATH):

    dataset = h5py.File(PATH, "r")
    train_images = dataset["train_img"]
    train_images = np.array(train_images, dtype=np.float32)
    val_images = dataset["val_img"]
    val_images = np.array(val_images, dtype=np.float32)
    test_images = dataset["test_img"]
    test_images = np.array(test_images, dtype=np.float32)

    train_images = scaler(train_images, (-1, 1))
    val_images = scaler(val_images, (-1, 1))
    test_images = scaler(test_images, (-1, 1))

    return train_images, val_images, test_images


def scaler(X, range):
    X_std = (X - np.min(X))/(np.max(X) - np.min(X))
    X_scaled = X_std * (range(1) - range(0)) + range(0)
    return X_scaled

