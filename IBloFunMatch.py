# Functions for computing and plotting the induced block function and matchings 
import sys 
import os
from token import NEWLINE 
import numpy as np
import scipy.spatial.distance as dist
import random


def main(perc, data_file):
    # Check correct percentage
    assert((perc>=0) and (perc<100)) 
    # Read input point cloud and compute distance matrix 
    points = np.genfromtxt(data_file)
    # Take "perc" indices of samples and store into file 
    random.seed(0)
    indices_sample = random.sample(range(points.shape[0]), int(points.shape[0]*perc))
    np.savetxt("output/indices_sample.out", indices_sample, fmt="%d", newline="\n")
    # Compute distance matrix of X and store 
    Dist_X = dist.squareform(dist.pdist(points))
    np.savetxt("output/dist_X.out", Dist_X, fmt="%.18e", delimiter=" ", newline="\n")
    # Compute distance matrix of sample and store 
    Dist_S = dist.squareform(dist.pdist(points[indices_sample]))
    np.savetxt("output/dist_S.out", Dist_S, fmt="%.18e", delimiter=" ", newline="\n")


if __name__== "__main__":
    perc = float(sys.argv[1])
    data_file = sys.argv[2]
    main(perc, data_file)
