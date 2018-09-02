# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import *
from utilities import load_h5py_datasets
import argparse

# parse some arguments
parser = argparse.ArgumentParser(description='Predict real qsm data using the trained model')
parser.add_argument('--pwrap_path', type=str,
                    help='Specify the path of the pwrap dataset')
parser.add_argument('--unwrap_path', type=str,
                    help='Specify the path of the orig dataset')
parser.add_argument('--weight_path', type=str,
                    help='Specify the path for the weights to be saved')
parser.add_argument('--save_path', type=str,
                    help='Specify the path for the results to be saved')
args = parser.parse_args()

# define some useful parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
WEIGHT_DIR = args.weight_path  # '/home/563/ls1729/gdata/phase_unwrapping/weights/PU_unet_002.hdf5'
PWRAP_DATASET_PATH = args.pwrap_path  # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/pwrap/pwrap_dataset.hdf5'
UNWRAP_DATASET_PATH = args.unwrap_path
SAVE_PATH = args.save_path  # '/home/563/ls1729/gdata/phase_unwrapping/real_qsm_samples/sample.jpg'

# load the datasets
print('Loading the Datasets...')

pwrap_dataset = h5py.File(PWRAP_DATASET_PATH, "r")
pwrap_images = pwrap_dataset["img"]
pwrap_images = np.array(pwrap_images, dtype=np.float32)

unwrap_dataset = h5py.File(UNWRAP_DATASET_PATH, "r")
unwrap_images = unwrap_dataset["img"]
unwrap_images = np.array(unwrap_images, dtype=np.float32)

print('Making predictions on real qsm data...')

# Predict on train, val and test
model = load_model(WEIGHT_DIR)
pred_images = model.predict(pwrap_images[80:91], verbose=1)

# compute the error


ix = random.randint(0, np.size(pred_images, 0)-1)

im_pwrap = np.reshape(pwrap_images[ix], (IMG_HEIGHT, IMG_WIDTH))
im_pred = np.reshape(pred_images[ix], (IMG_HEIGHT, IMG_WIDTH))
im_unwrap = np.reshape(unwrap_images[ix], (IMG_HEIGHT, IMG_WIDTH))

print('Saving Results...')

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax1, ax2, ax3 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_pred, cmap='gray'), ax=ax2)
ax2.set_title('Unwrapped Image - UNET')
fig.colorbar(ax3.imshow(im_unwrap, cmap='gray'), ax=ax3)
ax3.set_title('Unwrapped Image - LM')
plt.savefig(SAVE_PATH)


print('Complete..!')
