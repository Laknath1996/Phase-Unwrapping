# import libraries
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import *

def sample_predict(WEIGHT_DIR, pwrap_train_images, pwrap_val_images, pwrap_test_images, IMG_HEIGHT, IMG_WIDTH):

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
