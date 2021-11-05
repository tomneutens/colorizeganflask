import torch
import matplotlib.pyplot as plt

from torchvision import transforms

import data
import utils

import time

if __name__ == '__main__':
    # TODO: Data op USB stick
    #sets = data.construct_sets('/Volumes/Naamloos/Undithered', '../data/splits.yaml')
    device = torch.device('cpu')

    filenames = [
        '/Users/simon/Desktop/039-lieve-choco-gray-040.jpg'
       # '/Users/simon/Desktop/Schermafbeelding 2019-03-04 om 14.37.54.png',
       # '/Users/simon/Desktop/248-fifi-kampioen-color-018-uncolored.png'
    ]
    test_data_set = data.ImageDataset(filenames, transform=transforms.Compose([
        #transforms.RandomCrop(512),
        transforms.ToTensor(),
        utils.RgbToLab(),
        utils.AppendLineArt([35, 35])]))
    test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=1, shuffle=False)

    comparison_images = []

    for original, line_art in test_loader:
        line_art_original = line_art[0, 0].numpy()

        start = time.time()
        for _ in range(5):
            line_art = utils.remove_dithering(line_art_original)
        end = time.time()

        plt.imshow(line_art)
        plt.show()
        #io.imsave('test-new.png', color.grey2rgb(1 - line_art))
        print('it took', end - start)



