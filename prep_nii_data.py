# author : Ashwin de Silva
# prepare .nii dataset

# import libraries
import nibabel as nib
import numpy as np
import h5py
import argparse
import cv2

# pass some arguments
parser = argparse.ArgumentParser(description='Create training, validating and testing datasets with desired phase wraps.')
parser.add_argument('--nii_path', type=str,
                    help='Specify the path for the .nii image dataset. These images are fresh and not intenisty rescaled.')
parser.add_argument('--nat_path', type=str,
                    help='Specify the path for the images with wraps would be saved')

args = parser.parse_args()

# define some useful parameters
NII_DATASET_PATH = args.nii_path
NATURAL_DATASET_PATH = args.nat_path
SIZE = (512, 512)

# print the args
print('NII_DATASET_PATH : ', NII_DATASET_PATH)
print('NATURAL_DATASET_PATH : ', NATURAL_DATASET_PATH)
print('\n')

# load the .nii images
print('Loading the .nii data...')
img = nib.load(NATURAL_DATASET_PATH)
data = img.get_fdata()
data = np.array(data)

# select 30 images
data = data[:, :, 500:531]

# split the data to train, val and test
train_data = data[:, :, :20]
val_data = data[:, :, 20:25]
test_data = data[:, :, 25:30]

# define the dataset shape
train_shape = (np.size(train_data, 2), SIZE[0], SIZE[1], 1)
val_shape = (np.size(val_data, 2), SIZE[0], SIZE[1], 1)
test_shape = (np.size(test_data, 2), SIZE[0], SIZE[1], 1)

# open a hdf5 file and create earrays
dataset_file = h5py.File(NATURAL_DATASET_PATH, mode='w')

dataset_file.create_dataset("train_img", train_shape, np.int8)
dataset_file.create_dataset("val_img", val_shape, np.int8)
dataset_file.create_dataset("test_img", test_shape, np.int8)

# loop over train addresses
for i in range(train_shape[0]):
    # print how many images are saved
    print('Train data: {}/{}'.format(i+1, train_shape[0]))
    # read an image and resize
    img = train_data[:, :, i]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    dataset_file["train_img"][i, ...] = img[None]

# loop over validation addresses
for i in range(val_shape[0]):
    # print how many images are saved every 1000 images
    print('Validation data: {}/{}'.format(i, val_shape[0]))
    # read an image and resize
    img = val_data[:, :, i]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    dataset_file["val_img"][i, ...] = img[None]

# loop over the test addresses
for i in range(test_shape[0]):
    # print how many images are saved every 1000 images
    print('Test data: {}/{}'.format(i, test_shape[0]))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = test_data[:, :, i]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    # save the image
    dataset_file["test_img"][i, ...] = img[None]

dataset_file.close()

