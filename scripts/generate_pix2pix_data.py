""" Converts comic images to the A (original) and B (edges) images expected by the PyTorch pix2pix implementation. """
import torch

from torchvision import datasets, transforms
from skimage import color, io

# TODO: Fix this hackery by learning proper Python imports
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utils

if __name__ == '__main__':
    dataset = datasets.ImageFolder('/Users/simon/Desktop/jommeke/training/',
                                   transform=transforms.Compose([transforms.ToTensor(), utils.RgbToLab()]))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for batch_idx, (original, _) in enumerate(loader):
        print(batch_idx, '/', len(loader))
        original = original.numpy()
        line_art = utils.convert_lab_to_line_art(original)

        original_rgb = color.lab2rgb(original[0].transpose((1, 2, 0)) * [100, 128, 128])
        line_art_rgb = color.gray2rgb(1 - line_art[0, 0])

        io.imsave('A/' + str(batch_idx) + '.png', original_rgb)
        io.imsave('B/' + str(batch_idx) + '.png', line_art_rgb)
