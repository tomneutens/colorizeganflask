import numpy as np
import torch
import scipy
import scipy.stats
import cv2

import torch.nn as nn

#from numba import jit
from typing import List, Tuple


def convert_lab_to_line_art_single_image(image, precision=[25, 25, 25]):
    """ Input: A NumPy array of N x 3 x W x H in CIELab (normalized between 0 and 1)
        Output: A NumPy array of N x 1 x W x H where each number is either 0 or 1. Note that a 0 denotes the background,
        while a 1 denotes a line. """
    l = image[0, :, :] * 100
    a = image[1, :, :] * 128
    b = image[2, :, :] * 128

    out = scipy.stats.norm.pdf(l, 0, precision[0]) * scipy.stats.norm.pdf(a, 0, precision[1]) * scipy.stats.norm.pdf(b, 0, precision[2])
    out /= np.max(out)
    out[out >= 0.3] = 1
    out[out < 0.3]  = 0

    return out[np.newaxis, :, :]


def convert_lab_to_line_art(images, precision=[25, 25, 25]):
    """ Input: A NumPy array of N x 3 x W x H in CIELab (normalized between 0 and 1)
        Output: A NumPy array of N x 1 x W x H where each number is either 0 or 1. Note that a 0 denotes the background,
        while a 1 denotes a line. """
    l = images[:, 0, :, :] * 100
    a = images[:, 1, :, :] * 128
    b = images[:, 2, :, :] * 128

    out = scipy.stats.norm.pdf(l, 0, precision[0]) * scipy.stats.norm.pdf(a, 0, precision[1]) * scipy.stats.norm.pdf(b, 0, precision[2])
    out /= np.max(out)
    out[out >= 0.3] = 1
    out[out < 0.3]  = 0

    return out[:, np.newaxis, :, :]


def roll(tensor, shift, axis):
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def image_gradient_loss(img, device, line_art=None):
    """  Input should be a normalized (batch_size, 3, H, W) set of Lab images """
    # Construct a mask for the pixels that are on an lines
    # Note that we smudge the mask to make sure that pixels on the edge of a line are also ignored
    if line_art is not None:
        mask_edges = line_art[:, 0, :, :].to(torch.uint8)
    else:
        mask_edges = (img[:, 0, :, :] < (25 / 100)) & (img[:, 1, :, :] < (25 / 128)) & (img[:, 2, :, :] < (25 / 128))

    mask_edges = roll(mask_edges, -3, 1) | roll(mask_edges, -2, 1) | roll(mask_edges, -1, 1) | mask_edges | roll(mask_edges, 1, 1) | roll(mask_edges, 2, 1) | roll(mask_edges, 3, 1)
    mask_edges = roll(mask_edges, -3, 2) | roll(mask_edges, -2, 2) | roll(mask_edges, -1, 2) | mask_edges | roll(mask_edges, 1, 2) | roll(mask_edges, 2, 2) | roll(mask_edges, 3, 2)

    # Construct a Sobel filter for the horizontal direction
    kernel_x = np.tile(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), (3, 1, 1, 1))
    conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3).to(device)
    conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float().to(device), requires_grad=False)

    # Construct a Sobel filter for the vertical direction
    kernel_y = np.tile(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), (3, 1, 1, 1))
    conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3).to(device)
    conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float().to(device), requires_grad=False)

    # Compute the total gradient at each point in the image. Note that we ignore too large gradients since such abrupt
    # changes may be desired plus we don't want them to reduce the influence of the other smaller gradients.
    grad = torch.sum(torch.abs(conv_x(img)) + torch.abs(conv_y(img)), dim=1)
    grad = torch.tanh(grad)

    # Mask the computed gradient with the mask of edges to only account for the gradient apart from edges
    # Note that we also crop the resulting image to remove edge effects
    grad_masked = grad * (1 - mask_edges.float())
    grad_masked = grad_masked[:, 2:-2, 2:-2]

    grad_loss = grad_masked.mean()
    return grad_loss


def rgb_to_normalized_lab(image):
    # We originally used the function `skimage.color.rgb2lab` to perform the conversion. There is, however, an open
    # issue with this function being terrible slow. Consequently, we replaced it with the equivalent OpenCV function.
    # Source: https://github.com/scikit-image/scikit-image/issues/1133
    return cv2.cvtColor(image, cv2.COLOR_RGB2Lab) / [100, 128, 128]


#@jit(nopython=True)
def find_connected_component(img, x: int, y: int, visited) -> List[Tuple[int, int]]:
    """
    Determines the connected component (i.e. all connected filled-in pixels) for the given pixel.

    :param img: H x W binary NumPy array of 0's and 1's where 0 denotes a blank space and 1 denotes a line
    :param visited: H x W binary NumPy array of 0's and 1's where 1 denotes a position we've previously visited
    :return: A list of (x, y) pairs denoting each
    """
    to_visit, connected_pixels = [(x, y)], []

    while to_visit:
        x, y = to_visit.pop()
        if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0] or img[y, x] == 0 or visited[y, x]:
            continue

        connected_pixels.append((x, y))
        visited[y, x] = True

        for d_x, d_y in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            to_visit.append((x + d_x, y + d_y))

    return connected_pixels


#@jit(nopython=True)
def remove_dithering(img):
    """
    :param img: H x W binary NumPy arrays of 0's and 1's where 0 denotes a blank space and 1 denotes a line
    :return: Same as input but with dithering points cleared
    """
    height, width = img.shape

    output = img.copy()
    visited = np.full(img.shape, False)

    for x in range(width):
        for y in range(height):
            # This branch isn't necessary for correct execution but greatly increases performance by reducing the number
            # of function calls
            if visited[y, x] or img[y, x] == 0:
                continue

            connected_pixels = find_connected_component(img, x, y, visited)

            if len(connected_pixels) > 6:
                continue

            for x_, y_ in connected_pixels:
                output[y_, x_] = 0

    return output


# TODO: Move to package 'transforms'

class RgbToLab:

    def __call__(self, sample):
        image = sample.numpy().transpose((1, 2, 0))
        image = rgb_to_normalized_lab(image)
        return image.transpose((2, 0, 1))


class AppendLineArt:

    def __init__(self, lightness_scale):
        """
        :param lightness_scale: Either a single number between 0 and 100 or a tuple consisting of two numbers in that
        range. This decides the lightness cutoff point for the line art conversion. If a tuple is given, we randomly
        sample uniformly from the interval specified by the tuple.
        """
        self.lightness_scale = lightness_scale

    def __call__(self, sample):
        if type(self.lightness_scale) is tuple or type(self.lightness_scale) is list:
            lightness_scale = np.random.uniform(self.lightness_scale[0], self.lightness_scale[1])
        else:
            lightness_scale = self.lightness_scale

        line_art = torch.as_tensor(convert_lab_to_line_art_single_image(sample, precision=[lightness_scale, 25, 25]), dtype=torch.float)
        return sample, line_art
