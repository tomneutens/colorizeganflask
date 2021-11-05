import numpy as np
import torch
from scipy import ndimage

import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb
from skimage import io, color

import utils

filename = "/Users/simon/Desktop/248-fifi-kampioen-color-018-original.png"
# Keep only the lightness channel of the Lab image
img = (rgb2lab(io.imread(filename)).transpose((2, 0, 1)) / np.array([100, 128, 128])[:, np.newaxis, np.newaxis])[np.newaxis, :, :, :]
img = utils.convert_lab_to_line_art(img, precision=[25, 25, 25]).astype('bool')


neighbors_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

height = img.shape[2]
width = img.shape[3]

print('start')
res = ndimage.convolve(img[0, 0].astype('float64'), neighbors_kernel) #todo check min and max
res *= (res > 0) * np.random.uniform(size=res.shape)
res = res > 0.8
img |= res

res = ndimage.convolve(img[0, 0].astype('float64'), neighbors_kernel) #todo check min and max
res = (res == 1) * np.random.uniform(size=res.shape)
res = res > 0.8
img |= res
print('end')

# # TODO: Make more efficient by parallelizing
# print('start')
# for row in range(1, height - 1):
#     for col in range(1, width - 1):
#         neighbours = img[0, 0, row - 1, col] | img[0, 0, row + 1, col] | img[0, 0, row, col - 1] | img[0, 0, row, col + 1]
#         if neighbours:
#             img[0, 0, row, col] |= np.random.uniform() > 0.8
# print('end')

#uniform_noise = np.random.uniform(size=img.shape) > 0.99
#print(uniform_noise)
#img |= uniform_noise

img = img.astype('float64')
line_art_rgb = color.gray2rgb(1 - img[0, 0])
io.imsave('gray.png', line_art_rgb)


plt.imshow(line_art_rgb)
plt.show()