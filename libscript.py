from __future__ import print_function
import torch

from unet.unet_model import UNet
from tqdm import tqdm
import os

import numpy as np

from collections import defaultdict
from PIL import Image
from skimage import color, io
from torch.utils.data import DataLoader
from torchvision import transforms

import data
import utils

if __name__ == '__main__':

    no_cuda = True
    use_cuda = torch.cuda.is_available() and not no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')

    print('running inference on ' + device.__str__())

    model = UNet(1, 3).to(device)

    model.load_state_dict(torch.load('./models/netG-l1-discriminator-total-variation.torch', map_location=device))

    model.eval()

    # load data
    assigned_set = 'validation'
    filename = './images/imposters/fck05.jpeg'
    #filename = './images/137-apen-te-koop-color-005.jpg'
    #filename = './images/002-de-zingende-aap-gray-022.jpg'
    #filename = './images/001-de-jacht-op-een-voetbal-color-011.jpg'
    #filename = './images/part_jacht_voetbal_009.png'
    #filename = './images/127-de-watervallen-van-svanjabak-color-022.jpg'
    #filename = './images/002-de-zingende-aap-gray-015.jpg'
    #filename = './images/169-de-slangegodin-color-004.jpg'
    #filename = './images/139-de-blauwe-grot-color-040.jpg'
    #filename = './images/145-het-levenselixir-color-004.jpg'

    comics = defaultdict(set)
    sets = defaultdict(set)
    sets[assigned_set].add(filename)
    sets = dict(sets)
    #data_set = data.ImageDataset(sets[assigned_set],
    #                             transform=transforms.Compose([transforms.CenterCrop([1080*1500/960,1500]),transforms.ToTensor(), utils.RgbToLab()]))
    data_set = data.ImageDataset(sets[assigned_set],
                                 transform=transforms.Compose(
                                     [transforms.CenterCrop([700, 500]), transforms.ToTensor(),
                                      utils.RgbToLab()]))
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)

    # write result data
    try:
        os.makedirs('output')
    except FileExistsError:
        pass

    # calculate result data
    with torch.no_grad():
        for filename, original in tqdm(zip(data_set.filenames, data_loader), total=len(data_loader)):
            base_name = os.path.splitext(os.path.split(filename)[1])[0]
            if os.path.isfile('output/' + base_name + '-original.png'):
                print('Ignoring file', filename, 'since it already exists')
                continue

            # Uncomment to enable downscaling of image that don't fit in memory
            # width = original.shape[2]
            # if width > 960:
            #      original = F.interpolate(original, scale_factor=960 / width, mode='bicubic')

            line_art = utils.convert_lab_to_line_art(original, [30, 25, 25])
            #if args.remove_dither:
            #    line_art = utils.remove_dithering(line_art[0, 0])[np.newaxis, np.newaxis]
            line_art = torch.as_tensor(line_art, dtype=torch.float, device=device)
            colored = model(line_art).cpu() if model else None

            original_rgb = color.lab2rgb(original[0].numpy().transpose((1, 2, 0)) * [100, 128, 128])
            colored_rgb = color.lab2rgb(colored[0].numpy().transpose((1, 2, 0)) * [100, 128, 128]) if model is not None else None
            line_art_rgb = color.gray2rgb(1 - line_art.cpu().numpy()[0, 0])

            # a = np.asarray((colored_rgb,line_art_rgb))
            a = np.hstack((line_art_rgb, colored_rgb))
            a = 255 * a
            a = a.astype(np.uint8)
            im = Image.fromarray(a)
            im.show()
