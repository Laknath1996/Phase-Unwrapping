# author : Ashwin de Silva
# some helper functions for the main modules

# import libraries

import cv2
import numpy as np
import tensorflow as tf


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


