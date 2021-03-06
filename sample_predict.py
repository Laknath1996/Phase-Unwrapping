# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import *
from utilities import load_h5py_datasets
import argparse
from skimage.restoration import unwrap_phase
from skimage import exposure
from utilities import *

# parse some arguments
parser = argparse.ArgumentParser(description='Predict sample using the trained model')
parser.add_argument('--pwrap_path', type=str,
                    help='Specify the path of the pwrap dataset')
parser.add_argument('--orig_path', type=str,
                    help='Specify the path of the orig dataset')
parser.add_argument('--weight_path', type=str,
                    help='Specify the path for the weights to be saved')
args = parser.parse_args()

# define some useful parameters
IMG_HEIGHT = 160
IMG_WIDTH = 160
WEIGHT_DIR = args.weight_path  # '/home/563/ls1729/gdata/phase_unwrapping/weights/PU_unet_002.hdf5'
PWRAP_DATASET_PATH = args.pwrap_path  # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/pwrap/pwrap_dataset.hdf5'
ORIG_DATASET_PATH = args.orig_path

# load the datasets
print('Loading the Datasets...')
pwrap_train_images, pwrap_val_images, pwrap_test_images = load_h5py_datasets(PWRAP_DATASET_PATH)
orig_train_images, orig_val_images, orig_test_images = load_h5py_datasets(ORIG_DATASET_PATH)

print('Making predictions on train, val and test sets...')

# Predict on train, val and test
model = load_model(WEIGHT_DIR)
preds_train = model.predict(pwrap_train_images[:10], verbose=1)
preds_val = model.predict(pwrap_val_images[:10], verbose=1)
preds_test = model.predict(pwrap_test_images[:10], verbose=1)

# compute the error
train_SSD_unet = compute_ssd(preds_train, orig_train_images[:10])
val_SSD_unet = compute_ssd(preds_val, orig_val_images[:10])
test_SSD_unet = compute_ssd(preds_test, orig_test_images[:10])

train_mean_diff = compute_diff(preds_train, orig_train_images[:10])
val_mean_diff = compute_diff(preds_val, orig_val_images[:10])
test_mean_diff = compute_diff(preds_test, orig_test_images[:10])

print('Train SSD - unet : ', train_SSD_unet)
print('Val SSD - unet : ', val_SSD_unet)
print('Test SSD - unet : ', test_SSD_unet)
print('Train diff - unet : ', train_mean_diff)
print('Train diff - unet : ', val_mean_diff)
print('Train diff - unet : ', test_mean_diff)

# get a random image to save
ix = random.randint(0, len(preds_test)-1)
train_results = np.zeros((5, IMG_WIDTH, IMG_HEIGHT))
train_results[0] = im_pwrap_train = np.reshape(pwrap_train_images[ix], (IMG_HEIGHT, IMG_WIDTH))
train_results[1] = im_res_train = np.reshape(preds_train[ix], (IMG_HEIGHT, IMG_WIDTH))
train_results[2] = im_orig_train = np.reshape(orig_train_images[ix], (IMG_HEIGHT, IMG_WIDTH))
train_results[3] = im_ctrl_train = np.reshape(unwrap_phase(exposure.rescale_intensity(im_pwrap_train, (-np.pi, np.pi))), (IMG_HEIGHT, IMG_WIDTH))
train_results[4] = train_diff = im_orig_train - im_res_train

val_results = np.zeros((5, IMG_WIDTH, IMG_HEIGHT))
val_results[0] = im_pwrap_val = np.reshape(pwrap_val_images[ix], (IMG_HEIGHT, IMG_WIDTH))
val_results[1] = im_res_val = np.reshape(preds_val[ix], (IMG_HEIGHT, IMG_WIDTH))
val_results[2] = im_orig_val = np.reshape(orig_val_images[ix], (IMG_HEIGHT, IMG_WIDTH))
val_results[3] = im_ctrl_val = np.reshape(unwrap_phase(exposure.rescale_intensity(im_pwrap_val, (-np.pi, np.pi))), (IMG_HEIGHT, IMG_WIDTH))
val_results[4] = val_diff = im_orig_val - im_res_val

test_results = np.zeros((5, IMG_WIDTH, IMG_HEIGHT))
test_results[0] = im_pwrap_test = np.reshape(pwrap_test_images[ix], (IMG_HEIGHT, IMG_WIDTH))
test_results[1] = im_res_test = np.reshape(preds_test[ix], (IMG_HEIGHT, IMG_WIDTH))
test_results[2] = im_orig_test = np.reshape(orig_test_images[ix], (IMG_HEIGHT, IMG_WIDTH))
test_results[3] = im_ctrl_test = np.reshape(unwrap_phase(exposure.rescale_intensity(im_pwrap_test, (-np.pi, np.pi))), (IMG_HEIGHT, IMG_WIDTH))
test_results[4] = test_diff = im_orig_test - im_orig_test

print('Saving Results...')

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_train, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_train, cmap='gray'), ax=ax2)
ax2.set_title('unet Unwrapped Image')
fig.colorbar(ax3.imshow(im_orig_train, cmap='gray'), ax=ax3)
ax3.set_title('Original Image')
fig.colorbar(ax4.imshow(im_ctrl_train, cmap='gray'), ax=ax4)
ax4.set_title('py Unwrapped Image')
plt.show()
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/train_sample.jpg', bbox_inches='tight')

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_val, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_val, cmap='gray'), ax=ax2)
ax2.set_title('unet Unwrapped Image')
fig.colorbar(ax3.imshow(im_orig_val, cmap='gray'), ax=ax3)
ax3.set_title('Original Image')
fig.colorbar(ax4.imshow(im_ctrl_val, cmap='gray'), ax=ax4)
ax4.set_title('py Unwrapped Image')
plt.show()
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/val_sample.jpg', bbox_inches='tight')

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
ax1, ax2, ax3, ax4 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_test, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_test, cmap='gray'), ax=ax2)
ax2.set_title('unet Unwrapped Image')
fig.colorbar(ax3.imshow(im_orig_test, cmap='gray'), ax=ax3)
ax3.set_title('Original Image')
fig.colorbar(ax4.imshow(im_ctrl_test, cmap='gray'), ax=ax4)
ax4.set_title('py Unwrapped Image')
plt.show()
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/test_sample.jpg', bbox_inches='tight')

write_images(train_results, 1)
write_images(val_results, 2)
write_images(test_results, 3)

print('Complete..!')
