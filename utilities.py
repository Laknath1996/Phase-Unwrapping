# author : Ashwin de Silva
# some helper functions for the main modules

# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import h5py
import argparse


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

    # train_images = scaler(train_images, (-1, 1))
    # val_images = scaler(val_images, (-1, 1))
    # test_images = scaler(test_images, (-1, 1))

    return train_images, val_images, test_images


def scaler(X, range):
    X_std = (X - np.min(X))/(np.max(X) - np.min(X))
    X_scaled = X_std * (range[1] - range[0]) + range[0]
    return X_scaled

def get_args():
    parser = argparse.ArgumentParser(description='Train the unet architecture')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Specify the batch size')
    parser.add_argument('--loss', type=str, default='mean_squared_error',
                        help='Specify the path loss function')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Specify the number of epochs')
    parser.add_argument('--pwrap_path', type=str,
                        help='Specify the path of the pwrap dataset')
    parser.add_argument('--orig_path', type=str,
                        help='Specify the path of the orig dataset')
    parser.add_argument('--weight_path', type=str,
                        help='Specify the path for the weights to be saved')

    args = parser.parse_args()

    return args


def write_images(data, flag):
    n = np.size(data, 0)

    if flag == 1 :
        dict = {0:'train_phase_wrapped', 1:'train_unet_unwrapped', 2:'train_original', 3:'train_py_unwrapped', 4:'train_diff'}

    if flag == 2 :
        dict = {0:'val_phase wrapped', 1:'val_unet_unwrapped', 2:'val_original', 3:'val_py_unwrapped', 4:'val_diff'}

    if flag == 3 :
        dict = {0:'test_phase wrapped', 1:'test_unet_unwrapped', 2:'test_original', 3:'test_py_unwrapped', 4:'test_diff'}

    for i in range(n):
        im = data[i]
        plt.imsave(dict[i], im, cmap='gray')


def compute_ssd(X, Y):
    return np.mean(np.sum(np.square(X-Y), axis=(1, 2)))


def compute_diff(X,Y):
    return np.mean(np.sum(X-Y, axis=(1, 2)))

