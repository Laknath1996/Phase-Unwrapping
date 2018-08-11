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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard
import keras.backend as K
K.set_image_data_format('channels_last')
import os
from skimage import io
from skimage import data, img_as_float, color, exposure
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

#Define Parameters
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 1
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LOSS = 'binary_crossentropy'
STEPS_PER_EPOCH = 100
EPOCHS = 5
IMAGE_DIR = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/pwrap/'
MASK_DIR = '/home/563/ls1729/gdata/phase_unwrapping/dataset/coco/orig/'
WEIGHT_DIR = '/home/563/ls1729/gdata/phase_unwrapping/PU_unet_001.hdf5'
TB_LOG_DIR = '/home/563/ls1729/gdata/phase_unwrapping/logs'


# create two instances with the same arguments
print('Loading the Datasets...')
data_gen_args = dict(rescale=1./255,
                     validation_split=0.2,
                     featurewise_center=False,
                     featurewise_std_normalization=False)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 1
image_generator = image_datagen.flow_from_directory(
    IMAGE_DIR,
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    batch_size=BATCH_SIZE)

mask_generator = mask_datagen.flow_from_directory(
    MASK_DIR,
    target_size=(IMG_HEIGHT,IMG_WIDTH),
    color_mode='grayscale',
    class_mode=None,
    seed=seed,
    batch_size=BATCH_SIZE)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

# sketch the model
inputs = Input((512, 512, 1))#this should be (512,512,1)

conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2DTranspose(512, (2,2), strides = (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
#up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6],axis=3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

up7 = Conv2DTranspose(256, (2,2), strides = (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
#up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7],axis=3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

up8 = Conv2DTranspose(128, (2,2), strides = (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
#up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8],axis=3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

up9 = Conv2DTranspose(64, (2,2),  strides = (2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
#up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9],axis=3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(inputs = inputs, outputs = conv10)

# train the model
print('Fitting Model...')
model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = LOSS, metrics = ['accuracy'])
model.summary()
model_checkpoint = ModelCheckpoint(WEIGHT_DIR, monitor='loss', verbose=1, save_best_only=True)
tb = TensorBoard(log_dir=TB_LOG_DIR, batch_size=BATCH_SIZE, write_graph=True, write_images=True)
model.fit_generator(
    train_generator,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[model_checkpoint, tb])

#predict
print('Making Predictions on the Test Set...')
images = io.imread_collection('dataset/coco/test/*.jpg')
out_path = 'dataset/coco/results'
X_test = np.zeros(len(images.files), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
n = 0
for fn in images.files:
    img = io.imread(fn)[:,:,:IMG_CHANNELS]
    X_test[n] = img
    n += 1

model = load_model(WEIGHT_DIR)
preds_test = model.predict(X_test, verbose=1)
preds_test = (preds_test*255).astype(np.uint8)

for i in range(0,len(preds_test)):
    file = os.path.split(image.files[i])[1]
    path = os.path.join(out_path, file)
    plt.imsave(path, np.squeeze(preds_test[i]), cmap='gray')

print('Complete..!') 










