#author : Ashwin de Silva
# U-Net based phase unwrapping model

# import libraries
print('Importing the Libraries..')
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, concatenate, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.models import *
from keras.layers import merge, UpSampling2D,Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
K.set_image_data_format('channels_last')
import os
from skimage import io
from skimage import data, img_as_float, color, exposure
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from utilities import *
from unet_model import unet
import argparse
import random

# parse some useful parameters
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

#Define Parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1
BATCH_SIZE = args.batch_size
LEARNING_RATE = 1e-4
LOSS = args.loss
EPOCHS = args.epochs
PWRAP_DATASET_PATH = args.pwrap_path # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/pwrap/pwrap_dataset.hdf5'
ORIGINAL_DATASET_PATH = args.orig_path # '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/orig_dataset.hdf5'
WEIGHT_DIR = args.weight_path # '/home/563/ls1729/gdata/phase_unwrapping/weights/PU_unet_002.hdf5'
TB_LOG_DIR = '/home/563/ls1729/gdata/phase_unwrapping/logs'
PATIENCE = 3

print('Loading the Datasets...')

# load the datasets from the h5py files
orig_train_images, orig_val_images, orig_test_images = load_h5py_datasets(ORIGINAL_DATASET_PATH)
pwrap_train_images, pwrap_val_images, pwrap_test_images = load_h5py_datasets(PWRAP_DATASET_PATH)

STEPS_PER_EPOCH = np.size(orig_train_images, 0)/BATCH_SIZE
VALIDATION_STEPS = np.size(orig_val_images, 0)/BATCH_SIZE

# create the image generator objects
orig_datagen = ImageDataGenerator()
pwrap_datagen = ImageDataGenerator()
orig_datagen_val = ImageDataGenerator()
pwrap_datagen_val = ImageDataGenerator()

seed = 1 # keep the same seed for consistency

orig_datagen.fit(orig_train_images, augment=False, seed=seed)
pwrap_datagen.fit(pwrap_train_images, augment=False, seed=seed)
orig_datagen_val.fit(orig_val_images, augment=False, seed=seed)
pwrap_datagen_val.fit(pwrap_val_images, augment=False, seed=seed)

X = pwrap_datagen.flow(pwrap_train_images, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
Y = orig_datagen.flow(orig_train_images, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
X_val = pwrap_datagen.flow(pwrap_val_images, batch_size=BATCH_SIZE, shuffle=True, seed=seed)
Y_val = orig_datagen.flow(orig_val_images, batch_size=BATCH_SIZE, shuffle=True, seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(X, Y)
val_generator = zip(X_val, Y_val)

# sketch the model
input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model = unet(input_shape)

# train the model
print('Fitting Model...')
model.compile(optimizer=Adam(lr=LEARNING_RATE), loss=LOSS, metrics=['accuracy'])
model.summary()
earlystopper = EarlyStopping(patience=PATIENCE, verbose=1)
model_checkpoint = ModelCheckpoint(WEIGHT_DIR, monitor='loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir=TB_LOG_DIR, batch_size=BATCH_SIZE, write_graph=True, write_images=True)
model.fit_generator(
    train_generator,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[model_checkpoint, tb, earlystopper])

# predict

print('Making predictions on train, val and test sets...')

# Predict on train, val and test
model = load_model(WEIGHT_DIR)
preds_train = model.predict(pwrap_train_images[:10], verbose=1)
preds_val = model.predict(pwrap_val_images[:10], verbose=1)
preds_test = model.predict(pwrap_test_images[:10], verbose=1)

ix = random.randint(0, len(preds_test)-1)

im_pwrap_train = np.reshape(pwrap_train_images[ix], (IMG_HEIGHT, IMG_WIDTH))
im_res_train = np.reshape(preds_train[ix], (IMG_HEIGHT, IMG_WIDTH))

im_pwrap_val = np.reshape(pwrap_val_images[ix], (IMG_HEIGHT, IMG_WIDTH))
im_res_val = np.reshape(preds_val[ix], (IMG_HEIGHT, IMG_WIDTH))

im_pwrap_test = np.reshape(pwrap_test_images[ix], (IMG_HEIGHT, IMG_WIDTH))
im_res_test = np.reshape(preds_test[ix], (IMG_HEIGHT, IMG_WIDTH))

print('Saving Results...')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_train, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_train, cmap='gray'),
             ax=ax2)
ax2.set_title('Unwrapped Image')
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/train_sample.jpg')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_val, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_val, cmap='gray'),
             ax=ax2)
ax2.set_title('Unwrapped Image')
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/val_sample.jpg')

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax1, ax2 = ax.ravel()
fig.colorbar(ax1.imshow(im_pwrap_test, cmap='gray'), ax=ax1)
ax1.set_title('Wrapped Image')
fig.colorbar(ax2.imshow(im_res_test, cmap='gray'),
             ax=ax2)
ax2.set_title('Unwrapped Image')
plt.savefig('/home/563/ls1729/gdata/phase_unwrapping/samples/test_sample.jpg')

print('Complete..!') 










