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
INTENSITY_RESCALE_WINDOW = (-10, 10)
SIZE = (512, 512)

# print the args
print('NATURAL_DATASET_PATH : ', NATURAL_DATASET_PATH)
print('PWRAP_DATASET_PATH : ', PWRAP_DATASET_PATH)
print('ORIGINAL_DATASET_PATH : ', ORIGINAL_DATASET_PATH)
print('INTENSITY_RESCALE_WINDOW : ',INTENSITY_RESCALE_WINDOW)
print('SIZE : ', SIZE)
print('\n')

# load the dataset
print('Loading the original dataset...')
train_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
train_images = train_dataset["train_img"]
train_images = np.array(train_images, dtype=np.float32)
train_shape = np.shape(train_images)

val_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
val_images = val_dataset["val_img"]
val_images = np.array(val_images, dtype=np.float32)
val_shape = np.shape(val_images)

test_dataset = h5py.File(NATURAL_DATASET_PATH, "r")
test_images = test_dataset["test_img"]
test_images = np.array(test_images, dtype=np.float32)
test_shape = np.shape(test_images)

# open a hdf5 file and create earrays
print('creating the dataset with phase wraps...')
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
for i in range(train_shape[0]):
    im = train_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if i % 500 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, train_shape[0]))

    # rescale the pixel values
    im = exposure.rescale_intensity(im, out_range=INTENSITY_RESCALE_WINDOW)
    wrap_im = np.angle(np.exp(1j * im))

    # normalize and reshape the images
    orig_im = exposure.rescale_intensity(im, out_range=(-1, 1))
    wrap_im = exposure.rescale_intensity(wrap_im, out_range=(-1, 1))
    wrap_im = np.reshape(im, (SIZE[0], SIZE[1], 1))
    orig_im = np.reshape(im, (SIZE[0], SIZE[1], 1))

    # save the images in the h5 file
    original_dataset_file["train_img"][i, ...] = orig_im[None]
    wrapped_dataset_file["train_img"][i, ...] = wrap_im[None]


print('saving the validation set with phase wraps')
for i in range(val_shape[0]):
    im = val_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if i % 500 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, val_shape[0]))

    # rescale the pixel values
    im = exposure.rescale_intensity(im, out_range=INTENSITY_RESCALE_WINDOW)
    wrap_im = np.angle(np.exp(1j * im))

    # normalize and reshape the images
    orig_im = exposure.rescale_intensity(im, out_range=(-1, 1))
    wrap_im = exposure.rescale_intensity(wrap_im, out_range=(-1, 1))
    wrap_im = np.reshape(im, (SIZE[0], SIZE[1], 1))
    orig_im = np.reshape(im, (SIZE[0], SIZE[1], 1))

    # save the images in the h5 file
    original_dataset_file["val_img"][i, ...] = orig_im[None]
    wrapped_dataset_file["val_img"][i, ...] = wrap_im[None]

print('saving the test set with phase wraps')
for i in range(test_shape[0]):
    im = test_images[i, :, :, :]
    im = np.reshape(im, SIZE)

    if i % 500 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, val_shape[0]))

    # rescale the pixel values
    im = exposure.rescale_intensity(im, out_range=INTENSITY_RESCALE_WINDOW)
    wrap_im = np.angle(np.exp(1j * im))

    # normalize and reshape the images
    orig_im = exposure.rescale_intensity(im, out_range=(-1, 1))
    wrap_im = exposure.rescale_intensity(wrap_im, out_range=(-1, 1))
    wrap_im = np.reshape(im, (SIZE[0], SIZE[1], 1))
    orig_im = np.reshape(im, (SIZE[0], SIZE[1], 1))

    # save the images in the h5 file
    original_dataset_file["test_img"][i, ...] = orig_im[None]
    wrapped_dataset_file["test_img"][i, ...] = wrap_im[None]

original_dataset_file.close()
print('Complete!')






