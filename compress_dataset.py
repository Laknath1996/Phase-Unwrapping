# author : Ashwin de Silva
# compress the datatset in to a hdf5 format

# import the libraries

import glob
import numpy as np
import h5py
import cv2

# define some useful parameters

IMAGES_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/images_1/*.jpg'
DATASET_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/natural_dataset_1.hdf5'
SIZE = (512, 512)

# read addresses and labels from the 'train' folder
addrs = glob.glob(IMAGES_PATH)

# split the datasets in to training (60%), validation (20%) and test (20%)
train_addrs = addrs[0:int(0.6*len(addrs))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]

# define the dataset shape
train_shape = (len(train_addrs), SIZE[0], SIZE[1], 1)
val_shape = (len(val_addrs), SIZE[0], SIZE[1], 1)
test_shape = (len(test_addrs), SIZE[0], SIZE[1], 1)

# open a hdf5 file and create earrays
dataset_file = h5py.File(DATASET_PATH, mode='w')

dataset_file.create_dataset("train_img", train_shape, np.int8)
dataset_file.create_dataset("val_img", val_shape, np.int8)
dataset_file.create_dataset("test_img", test_shape, np.int8)

# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 500 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
    # read an image and resize
    addr = train_addrs[i]
    img = cv2.imread(addr, 0)
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    dataset_file["train_img"][i, ...] = img[None]

# loop over validation addresses
for i in range(len(val_addrs)):
    # print how many images are saved every 1000 images
    if i % 500 == 0 and i > 1:
        print('Validation data: {}/{}'.format(i, len(val_addrs)))
    # read an image and resize
    addr = val_addrs[i]
    img = cv2.imread(addr, 0)
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    dataset_file["val_img"][i, ...] = img[None]

# loop over the test addresses
for i in range(len(test_addrs)):
    # print how many images are saved every 1000 images
    if i % 500 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]
    img = cv2.imread(addr, 0)
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    # save the image
    dataset_file["test_img"][i, ...] = img[None]

dataset_file.close()
