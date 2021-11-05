import os
import yaml

from PIL import Image
from collections import defaultdict

import torch.utils.data as data


class ImageDataset(data.Dataset):
    """A dataset based on a set of paths to image files and an optional transform."""

    def __init__(self, filenames, transform=None):
        self.filenames = list(filenames)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        print('\tImageDataset: Opening image', self.filenames[idx])
        # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(self.filenames[idx], 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
            print('\tImageDataset: Opening image', self.filenames[idx])
            # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(self.filenames[idx], 'rb') as f:
                img = Image.open(f).convert('RGB')
                return self.transform(img) if self.transform else img


def construct_sets(root, description_file_name):
    """
    Returns a dict that maps a set name to a list of image paths, based on a given set file.

    :param root:
    :param description_file_name:
    :return:
    """
    # Maps a comic ID to a tuple of page IDs and their corresponding filename
    comics = defaultdict(set)

    for filename in (p for p in os.listdir(root) if '-color-' in p and not p.startswith('.')):
        comic_id = int(filename.split('-')[0])
        page_id = int(filename.split('-')[-1].split('.')[0])
        comics[comic_id].add((page_id, os.path.join(root, filename)))

    # Take a copy to prevent the defaultdict behavior from masking errors
    comics = dict(comics)

    # Maps set to filenames
    sets = defaultdict(set)

    with open(description_file_name) as data_file:
        data = yaml.load(data_file)

    for comic in data['comics']:
        all_pages = comics[comic['id']]

        # Make sure every referenced page in the YAML actually exists
        assert set(page_id for page_id in comic['pages'].keys() if page_id != 'def').issubset(set(page_id for page_id, _ in all_pages))

        for page_id, filename in all_pages:
            assigned_set = comic['pages'].get(page_id, comic['pages']['def'])
            sets[assigned_set].add(filename)

    return dict(sets)
