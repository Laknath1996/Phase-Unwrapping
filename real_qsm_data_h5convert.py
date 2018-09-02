# author : Ashwin de Silva
# prepare .nii dataset

# import libraries
import nibabel as nib
import numpy as np
import h5py
import argparse
import cv2
from skimage import exposure
from skimage import transform

# pass some arguments
# parser = argparse.ArgumentParser(description='Create training, validating and testing datasets with desired phase wraps.')
# parser.add_argument('--nii_path', type=str,
#                     help='Specify the path for the .nii image dataset. These images are fresh and not intenisty rescaled.')
# parser.add_argument('--nat_path', type=str,
#                     help='Specify the path for the images with wraps would be saved')
#
# args = parser.parse_args()

# define some useful parameters
NII_DATASET_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/qsm_2016_data/phs_wrap.nii'
NATURAL_DATASET_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/qsm_2016_data/phs_wrap.hdf5'
SIZE = (512, 512)

# print the args
print('NII_DATASET_PATH : ', NII_DATASET_PATH)
print('NATURAL_DATASET_PATH : ', NATURAL_DATASET_PATH)
print('\n')

# load the .nii images
print('Loading the .nii data...')
img = nib.load(NII_DATASET_PATH)
data = img.get_fdata()
data = np.array(data)

# define the dataset shape
data_shape = (np.size(data, 2), SIZE[0], SIZE[1], 1)

# open a hdf5 file and create earrays
dataset_file = h5py.File(NATURAL_DATASET_PATH, mode='w')

dataset_file.create_dataset("img", data_shape, np.float32)

# loop over train addresses
for i in range(data_shape[0]):
    # print how many images are saved
    print('Train data: {}/{}'.format(i+1, data_shape[0]))
    # read an image and resize
    img = data[:, :, i]
    img = cv2.resize(img, SIZE, interpolation=cv2.INTER_CUBIC)
    img = exposure.rescale_intensity(img, out_range=(-1, 1))
    img = transform.rotate(img, 180)
    img = np.reshape(img, (SIZE[0], SIZE[1], 1))
    # add any image pre-processing here

    dataset_file["img"][i, ...] = img[None]

dataset_file.close()

