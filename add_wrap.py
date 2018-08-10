# author : Ashwin de Silva
# adding the phase wrap to the images

# import libraries
import os
from skimage import io
from skimage import data, img_as_float, color, exposure
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

# load all the images
images = io.imread_collection('dataset/coco/val2017/*.jpg')
out_path = 'dataset/coco/orig/images'

# add the phase wrap and save the images
for fn in images.files:
    im = color.rgb2gray(img_as_float(io.imread(fn)))
    im = resize(im, (512, 512))
    im = exposure.rescale_intensity(im, out_range=(-3*np.pi, 3 * np.pi))
    #im = np.angle(np.exp(1j * im))
    #im = np.arctan2(np.sin(im), np.cos(im))
    file = os.path.split(fn)[1]
    path = os.path.join(out_path, file)
    plt.imsave(path, im, cmap='gray')




