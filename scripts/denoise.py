''' Denoises a binary image '''
# See https://github.com/pmneila/PyMaxflow/blob/master/examples/binary_restoration.py
import maxflow
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt
import sys

img = imread(sys.argv[1])
if len(img.shape) > 2:
    img = img[:, :, 0]

# Create the graph.
g = maxflow.Graph[int](0, 0)
# Add the nodes.
nodeids = g.add_grid_nodes(img.shape)
# Add edges with the same capacities.
g.add_grid_edges(nodeids, 100)
# Add the terminal edges.
g.add_grid_tedges(nodeids, img, 255-img)

graph = g.get_nx_graph()

# Find the maximum flow.
g.maxflow()
# Get the segments.
sgm = g.get_grid_segments(nodeids)

# The labels should be 1 where sgm is False and 0 otherwise.
img2 = np.int_(np.logical_not(sgm))
# Show the result.
imsave('denoised.png', img2)
