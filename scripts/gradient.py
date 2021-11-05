from PIL import Image
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
from skimage.color import rgb2lab, lab2rgb
from skimage import io
import matplotlib.pyplot as plt

from utils import convert_lab_to_line_art, image_gradient_loss


device = torch.device('cpu')
colorized_path = "/Users/simon/Desktop/248-fifi-kampioen-color-018-colored.png"
img = torch.from_numpy((rgb2lab(io.imread(colorized_path)).transpose((2, 0, 1)) / np.array([100, 128, 128])[:, np.newaxis, np.newaxis])[np.newaxis, :, :, :])
img = img.float()

mask_img = torch.from_numpy(convert_lab_to_line_art(img))

print(image_gradient_loss(img, device, mask_img))


# def roll(tensor, shift, axis):
#     if shift == 0:
#         return tensor
#
#     if axis < 0:
#         axis += tensor.dim()
#
#     dim_size = tensor.size(axis)
#     after_start = dim_size - shift
#     if shift < 0:
#         after_start = -shift
#         shift = dim_size - abs(shift)
#
#     before = tensor.narrow(axis, 0, dim_size - shift)
#     after = tensor.narrow(axis, after_start, shift)
#     return torch.cat([after, before], axis)
#
# device = torch.device('cpu')
#
# colorized_path = "/Users/simon/Desktop/248-fifi-kampioen-color-018-colored.png"
#
# mask_img = torch.from_numpy((rgb2lab(io.imread(colorized_path)).transpose((2, 0, 1)) / np.array([100, 128, 128])[:, np.newaxis, np.newaxis])[np.newaxis, :, :, :])
# mask_img = torch.from_numpy(convert_lab_to_line_art(mask_img))
# print(mask_img.shape)
#
#
# # Keep only the lightness channel of the Lab image
# img = torch.from_numpy((rgb2lab(io.imread(colorized_path)).transpose((2, 0, 1)) / np.array([100, 128, 128])[:, np.newaxis, np.newaxis])[np.newaxis, :, :, :])
# img = img.float()
#
# # Construct a mask for the pixels that are on an lines
# # Note that we smudge the mask to make sure that pixels on the edge of a line are also ignored
# mask_edges = mask_img[:, 0, :, :].to(torch.uint8)
# print(mask_edges[0, 865, 200])
# mask_edges_2 = (img[:, 0, :, :] < (25 / 100)) & (img[:, 1, :, :] < (25 / 128)) & (img[:, 2, :, :] < (25 / 128))
# print(mask_edges_2[0, 865, 200])
# print(mask_edges_2)
# print('----')
# print(mask_edges.shape)
# print(mask_edges_2.shape)
# print(mask_edges.type())
# print(mask_edges_2.type())
#
# mask_edges = roll(mask_edges, -3, 1) | roll(mask_edges, -2, 1) | roll(mask_edges, -1, 1) | mask_edges | roll(mask_edges, 1, 1) | roll( mask_edges, 2, 1) | roll(mask_edges, 3, 1)
# mask_edges = roll(mask_edges, -3, 2) | roll(mask_edges, -2, 2) | roll(mask_edges, -1, 2) | mask_edges | roll(mask_edges, 1, 2) | roll( mask_edges, 2, 2) | roll(mask_edges, 3, 2)
#
# # Construct a Sobel filter for the horizontal direction
# kernel_x = np.tile(np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]), (3, 3, 1, 1))
# conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).to(device)
# conv_x.weight = nn.Parameter(torch.from_numpy(kernel_x).float().to(device), requires_grad=False)
#
# # Construct a Sobel filter for the vertical direction
# kernel_y = np.tile(np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]), (3, 3, 1, 1))
# conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False).to(device)
# conv_y.weight = nn.Parameter(torch.from_numpy(kernel_y).float().to(device), requires_grad=False)
#
# # Compute the total gradient at each point in the image. Note that we ignore too large gradients since such abrupt
# # changes may be desired plus we don't want them to reduce the influence of the other smaller gradients.
# grad = torch.sum(torch.abs(conv_x(img)) + torch.abs(conv_y(img)), dim=1)
# grad = torch.nn.functional.tanh(grad)
#
# # Mask the computed gradient with the mask of edges to only account for the gradient apart from edges
# # Note that we also crop the resulting image to remove edge effects
# grad_masked = grad * (1 - mask_edges.float())
# grad_masked = grad_masked[:, 2:-2, 2:-2]
#
# plt.imshow(grad_masked[0])
# # plt.savefig('original-colored-grad-loss-masked.pdf', dpi=500)
# plt.show()
# # # plt.imshow(grad_x[0]) #mask_non_edges[0] * grad_x[0])
# # # plt.show()
