import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from skimage import color

import data
import utils


if __name__ == '__main__':
    # TODO: Data op USB stick
    sets = data.construct_sets('/Volumes/Naamloos/Undithered', '../data/splits.yaml')
    device = torch.device('cpu')

    test_data_set = data.ImageDataset(sets['train'], transform=transforms.Compose([
        transforms.RandomCrop(512),
        transforms.ToTensor(),
        utils.RgbToLab(),
        utils.AppendLineArt([25, 35])]))
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=1, shuffle=False)

    comparison_images = []

    for original, line_art in test_loader:
        if len(comparison_images) > 3: break

        _, cluster_visualization = utils.within_cluster_standard_deviation_loss(original, line_art, original, torch.device('cpu'), visualize=True)

        original_rgb = color.lab2rgb(original[0].numpy().transpose((1, 2, 0)) * [100, 128, 128])
        line_art_rgb = color.gray2rgb(1 - line_art.cpu().numpy()[0, 0])

        comparison_images += [np.concatenate((line_art_rgb, original_rgb, cluster_visualization))]

    comparison_image = np.concatenate(comparison_images, axis=1)
    print(comparison_image.shape)

    plt.imshow(comparison_image)
    plt.show()

