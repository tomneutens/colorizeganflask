import argparse
import os
import skimage
import torch
import warnings
import torch

from skimage import color, io
from torchvision import transforms
from tqdm import tqdm

import numpy as np

import torch.nn.functional as F

import data
import utils

# Colorizes a list of given images
from unet.unet_model import UNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize a given set using the specified model')
    parser.add_argument('--data-root', type=str, required=True, help='the directory where all the image files are stored')
    parser.add_argument('--set-file', type=str, required=True, help='the file containing the description of the different sets')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--remove-dither', action='store_true', default=False, help='enables dither removal on grayscale images')
    parser.add_argument('--set', type=str, default=None, help='the name of the set to colorize')
    parser.add_argument('--model', type=str, default=None, help='the path to the model to use. If not supplied, the colorization step is skipped')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model:
        model = UNet(1, 3).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
    else:
        model = None


    sets = data.construct_sets(args.data_root, args.set_file)
    data_set = data.ImageDataset(sets[args.set], transform=transforms.Compose([transforms.ToTensor(), utils.RgbToLab()]))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

    try:
        os.makedirs('output')
    except FileExistsError:
        pass

    with torch.no_grad():
        for filename, original in tqdm(zip(data_set.filenames, data_loader), total=len(data_loader)):
            base_name = os.path.splitext(os.path.split(filename)[1])[0]
            if os.path.isfile('output/' + base_name + '-original.png'):
                print('Ignoring file', filename, 'since it already exists')
                continue

            # Uncomment to enable downscaling of image that don't fit in memory
            # width = original.shape[2]
            # if width > 2100
            #     original = F.interpolate(original, scale_factor=2100 / width, mode='bilinear')

            line_art = utils.convert_lab_to_line_art(original, [30, 25, 25])
            if args.remove_dither:
                line_art = utils.remove_dithering(line_art[0, 0])[np.newaxis, np.newaxis]
            line_art = torch.as_tensor(line_art, dtype=torch.float, device=device)
            colored = model(line_art).cpu() if model else None

            original_rgb = color.lab2rgb(original[0].numpy().transpose((1, 2, 0)) * [100, 128, 128])
            colored_rgb = color.lab2rgb(colored[0].numpy().transpose((1, 2, 0)) * [100, 128, 128]) if model is not None else None
            line_art_rgb = color.gray2rgb(1 - line_art.cpu().numpy()[0, 0])

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                io.imsave('output/' + base_name + '-original.png', original_rgb)
                if colored is not None:
                    io.imsave('output/' + base_name + '-colored.png', colored_rgb)
                io.imsave('output/' + base_name + '-uncolored.png', line_art_rgb)
