# author : Ashwin de Silva
# phase simulator

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase
import h5py

# useful params
PWRAP_DATASET_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/brain_7T/pwrap/pwrap_dataset_1.hdf5'
ORIGINAL_DATASET_PATH = '/home/563/ls1729/gdata/phase_unwrapping/dataset/brain_7T/orig/orig_dataset_1.hdf5'
ARB = 25

# load the dataset
orig_dataset = h5py.File(ORIGINAL_DATASET_PATH, "r")
orig_train_images = orig_dataset["train_img"]
orig_train_images = np.array(orig_train_images, dtype=np.float32)

pwrap_dataset = h5py.File(PWRAP_DATASET_PATH, "r")
pwrap_train_images = pwrap_dataset["train_img"]
pwrap_train_images = np.array(pwrap_train_images, dtype=np.float32)

size = (np.size(orig_train_images, 1), np.size(orig_train_images, 2))

# select an arbitrary image
im_orig = orig_train_images[ARB, :, :, :]
im_orig = np.reshape(im_orig, size)

im_pwrap = pwrap_train_images[ARB, :, :, :]
im_pwrap = np.reshape(im_pwrap, size)


# plot the results

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()

fig.colorbar(ax1.imshow(im_orig, cmap='gray'), ax=ax1)
ax1.set_title('Original Image')

fig.colorbar(ax2.imshow(im_pwrap, cmap='gray'),
             ax=ax2)
ax2.set_title('Wrapped phase Image')

plt.savefig('check.jpg')
