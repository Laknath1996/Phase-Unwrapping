# author : Ashwin de Silva
# adding the phase wrap to the images

# import libraries
import os
from skimage import io
from skimage import data, img_as_float, color, exposure
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import cv2
from utilities import *

# pass some arguments
parser = argparse.ArgumentParser(description='Create training, validating and testing datasets with desired phase wraps.')
parser.add_argument('--nat_path', type=str,
                    help='Specify the path for the natural image dataset. These images are fresh and not intenisty rescaled.')
parser.add_argument('--orig_path', type=str,
                    help='Specify the path for the which the intensity rescaled images with wraps would be saved')
parser.add_argument('--pwrap_path', type=str,
                    help='Specify the path for which the images with phase wraps would be saved')

args = parser.parse_args()

# define some useful parameters
NATURAL_DATASET_PATH = args.nat_path  # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/natural_dataset.hdf5'
PWRAP_DATASET_PATH = args.pwrap_path  # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/pwrap/pwrap_dataset.hdf5'
ORIGINAL_DATASET_PATH = args.orig_path
SIZE = (512, 512)

RANDOM_RESCALING = False
LOWER_BOUND_ZERO = True
INTENSITY_RESCALE_WINDOW = (-50, 50)
NO_WRAPS = 100
LOWER_BOUNDS = [-4.5, -5, -5.5, -6, -6.5, -7, -7.5, -8, -8.5, -9, -9.5, -10]
HIGHER_BOUNDS = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]

RESIZE = True
NEW_SIZE = (160, 160)

# print the args
print('NATURAL_DATASET_PATH : ', NATURAL_DATASET_PATH)
print('PWRAP_DATASET_PATH : ', PWRAP_DATASET_PATH)
print('ORIGINAL_DATASET_PATH : ', ORIGINAL_DATASET_PATH)
print('SIZE : ', SIZE)
print('RANDOM SCALING? :', RANDOM_RESCALING)
print('LOWER BOUND ZERO? :', LOWER_BOUND_ZERO)
if RANDOM_RESCALING:
    print('INTENSITY_RESCALE_WINDOW : ',INTENSITY_RESCALE_WINDOW)
    print('NUMBER OF WRAPS : ', NO_WRAPS)
else:
    print('LOWER BOUNDS : ', LOWER_BOUNDS)
    print('HIGHER BOUNDS : ', HIGHER_BOUNDS)
print('RESIZE? :', RESIZE)
if RESIZE:
    print('NEW SIZE', NEW_SIZE)
else:
    NEW_SIZE = SIZE
print('\n')

# load the dataset
print('Loading the original dataset...')
train_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
train_images = train_dataset["train_img"]
train_images = np.array(train_images, dtype=np.float32)

val_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
val_images = val_dataset["val_img"]
val_images = np.array(val_images, dtype=np.float32)

test_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
test_images = test_dataset["test_img"]
test_images = np.array(test_images, dtype=np.float32)

# define the scaling schemes
if RANDOM_RESCALING:
    # create a random set of rescaling windows
    HIGH = np.random.randint(1, INTENSITY_RESCALE_WINDOW[1]+1, NO_WRAPS)
    if LOWER_BOUND_ZERO:
        LOW = np.zeros((NO_WRAPS, ))
    else:
        LOW = np.random.randint(INTENSITY_RESCALE_WINDOW[0], 0, NO_WRAPS)
else:
    # create predertermined rescalings
    LOW = LOWER_BOUNDS
    HIGH = HIGHER_BOUNDS
    NO_WRAPS = len(LOW) * len(HIGH)

# open a hdf5 file and create earrays
print('creating the dataset with phase wraps...')

train_shape = (np.size(train_images, 0)*NO_WRAPS, NEW_SIZE[0], NEW_SIZE[1], 1)
val_shape = (np.size(val_images, 0)*NO_WRAPS, NEW_SIZE[0], NEW_SIZE[1], 1)
test_shape = (np.size(test_images, 0)*NO_WRAPS, NEW_SIZE[0], NEW_SIZE[1], 1)

original_dataset_file = h5py.File(ORIGINAL_DATASET_PATH, mode='w') # dataset with rescaled intensities
wrapped_dataset_file = h5py.File(PWRAP_DATASET_PATH, mode='w') # dataset with wrapped intensities

original_dataset_file.create_dataset("train_img", train_shape, np.float32)
original_dataset_file.create_dataset("val_img", val_shape, np.float32)
original_dataset_file.create_dataset("test_img", test_shape, np.float32)

wrapped_dataset_file.create_dataset("train_img", train_shape, np.float32)
wrapped_dataset_file.create_dataset("val_img", val_shape, np.float32)
wrapped_dataset_file.create_dataset("test_img", test_shape, np.float32)

# add the phase wrap to the images
print('saving the train set with phase wraps')
k = 0
for i in range(np.size(train_images, 0)):
    im = train_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if RESIZE:
        im = cv2.resize(im, NEW_SIZE, interpolation=cv2.INTER_CUBIC)

    #if i % 500 == 0 and i > 1:
    print('Train data: {}/{}'.format(i+1, train_shape[0]/NO_WRAPS))

    if RANDOM_RESCALING:
        for j in range(NO_WRAPS):
            # add wraps
            orig_im, wrap_im = wrap_images(im, LOW[j], HIGH[j], NEW_SIZE)

            # save the images in the h5 file
            original_dataset_file["train_img"][k, ...] = orig_im[None]
            wrapped_dataset_file["train_img"][k, ...] = wrap_im[None]
            k += 1
    else:
        for m in range(len(LOW)):
            for n in range(len(HIGH)):
                # add wraps
                orig_im, wrap_im = wrap_images(im, LOW[m], HIGH[n], NEW_SIZE)

                # save the images in the h5 file
                original_dataset_file["train_img"][k, ...] = orig_im[None]
                wrapped_dataset_file["train_img"][k, ...] = wrap_im[None]
                k += 1


print('saving the validation set with phase wraps')
k = 0
for i in range(np.size(val_images, 0)):
    im = train_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if RESIZE:
        im = cv2.resize(im, NEW_SIZE, interpolation=cv2.INTER_CUBIC)

    #if i % 500 == 0 and i > 1:
    print('Validation data: {}/{}'.format(i+1, val_shape[0]/NO_WRAPS))

    if RANDOM_RESCALING:
        for j in range(NO_WRAPS):
            # add wraps
            orig_im, wrap_im = wrap_images(im, LOW[j], HIGH[j], NEW_SIZE)

            # save the images in the h5 file
            original_dataset_file["val_img"][k, ...] = orig_im[None]
            wrapped_dataset_file["val_img"][k, ...] = wrap_im[None]
            k += 1
    else:
        for m in range(len(LOW)):
            for n in range(len(HIGH)):
                # add wraps
                orig_im, wrap_im = wrap_images(im, LOW[m], HIGH[n], NEW_SIZE)

                # save the images in the h5 file
                original_dataset_file["val_img"][k, ...] = orig_im[None]
                wrapped_dataset_file["val_img"][k, ...] = wrap_im[None]
                k += 1


print('saving the test set with phase wraps')
k = 0
for i in range(np.size(test_images, 0)):
    im = train_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if RESIZE:
        im = cv2.resize(im, NEW_SIZE, interpolation=cv2.INTER_CUBIC)

    #if i % 500 == 0 and i > 1:
    print('Test data: {}/{}'.format(i+1, test_shape[0]/NO_WRAPS))

    if RANDOM_RESCALING:
        for j in range(NO_WRAPS):
            # add wraps
            orig_im, wrap_im = wrap_images(im, LOW[j], HIGH[j], NEW_SIZE)

            # save the images in the h5 file
            original_dataset_file["test_img"][k, ...] = orig_im[None]
            wrapped_dataset_file["test_img"][k, ...] = wrap_im[None]
            k += 1
    else:
        for m in range(len(LOW)):
            for n in range(len(HIGH)):
                # add wraps
                orig_im, wrap_im = wrap_images(im, LOW[m], HIGH[n], NEW_SIZE)

                # save the images in the h5 file
                original_dataset_file["test_img"][k, ...] = orig_im[None]
                wrapped_dataset_file["test_img"][k, ...] = wrap_im[None]
                k += 1

original_dataset_file.close()
print('Complete!')






