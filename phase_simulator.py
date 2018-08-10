# author : Ashwin de Silva
# phase simulator

import numpy as np
from matplotlib import pyplot as plt
import cv2
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase

# create a random matrix between -10 and 10

np.random.seed(0)
size = [160, 160]
phase_image = np.random.randn(160, 160)
phase_image = exposure.rescale_intensity(phase_image, out_range=(-10, 10))

# introduce a wrap

wrapped_phase_image = np.arctan2(np.sin(phase_image), np.cos(phase_image))

# unwrap the phase from the scikit-image

unwrapped_phase_image = unwrap_phase(wrapped_phase_image)

# plot the results

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
ax1, ax2, ax3 = ax.ravel()

fig.colorbar(ax1.imshow(phase_image, cmap='gray'), ax=ax1)
ax1.set_title('Original')

fig.colorbar(ax2.imshow(wrapped_phase_image, cmap='gray'),
             ax=ax2)
ax2.set_title('Wrapped phase')

fig.colorbar(ax3.imshow(unwrapped_phase_image, cmap='gray'),
             ax=ax3)
ax3.set_title('Unwrapped phase')

plt.show()
