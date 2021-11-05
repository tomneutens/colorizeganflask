import csv
import os
import sys

from utils import rgb_to_normalized_lab, image_gradient_loss
from skimage import io
from skimage.measure import compare_ssim
import numpy as np
import torch
import skimage
import skimage.filters


def calculate_histogram_variance(image):
    hist, bins = np.histogram(image, bins=50, density=True)
    center = (bins[:-1] + bins[1:]) / 2

    mean = np.sum(hist * center) / np.sum(hist)
    return np.sum(hist * (center - mean) ** 2) / np.sum(hist)


def loss_mae(original, reconstruction):
    return np.abs((original - reconstruction)).mean()


def loss_mse(original, reconstruction):
    return np.square((original - reconstruction)).mean()


def loss_gradient_ratio(original, reconstruction):
    """ How large is the gradient of the reconstruction image relative to the original image? """
    # The images are both blurred to remove the influence of dithering dots
    original = skimage.filters.gaussian(original, sigma=3, multichannel=True)
    reconstruction = skimage.filters.gaussian(reconstruction, sigma=3, multichannel=True)

    original_torch = torch.from_numpy(original.transpose((2, 0, 1))[np.newaxis]).float()
    reconstruction_torch = torch.from_numpy(reconstruction.transpose((2, 0, 1))[np.newaxis]).float()
    return image_gradient_loss(reconstruction_torch, reconstruction_torch.device).item()\
           / image_gradient_loss(original_torch, original_torch.device).item()


def score_hist_stddev_ratio(original, reconstruction):
    return np.sqrt(calculate_histogram_variance(reconstruction)) / np.sqrt(calculate_histogram_variance(original))


def score_ssim(original, reconstruction):
    return compare_ssim(original, reconstruction, multichannel=True)


def get_colorization_pairs(root):
    ''' Tuples (orginal path, reconstured path) '''
    all_files = [] # Tuples of (original, reconstruced)

    for path, _, files in os.walk(root):
        for name in files:
            if not name.endswith('-original.png'): continue
            original_filename = os.path.join(path, name)
            reconstruction_filename = original_filename.replace('-original.png', '-colored.png')
            if os.path.isfile(reconstruction_filename):
                all_files.append((original_filename, reconstruction_filename))

    return all_files


if __name__ == '__main__':
    metrics = [loss_mae, loss_mse, score_hist_stddev_ratio, score_ssim, loss_gradient_ratio]

    if len(sys.argv) != 2:
        sys.exit('Expects a single argument that denotes the root directory in which to search pairs')

    root = sys.argv[1]
    pairs = get_colorization_pairs(root)
    metrics_results = []

    results_writer = csv.writer(sys.stdout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(['original_path', 'reconstruction_path'] + [metric.__name__ for metric in metrics])

    for (original_path, reconstruction_path) in pairs:
        original, reconstruction = rgb_to_normalized_lab(io.imread(original_path)), rgb_to_normalized_lab(io.imread(reconstruction_path))
        metrics_results.append([metric(original, reconstruction) for metric in metrics])
        results_writer.writerow([original_path, reconstruction_path] + metrics_results[-1])

    results_writer.writerow(['mean', 'mean'] + np.array(metrics_results).mean(axis=0).tolist())
    results_writer.writerow(['stddev', 'stddev'] + np.array(metrics_results).std(axis=0).tolist())
