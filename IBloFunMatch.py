# Functions for computing and plotting the induced block function and matchings 
import sys 
import os
# from token import NEWLINE 
import numpy as np
import scipy.spatial.distance as dist
import random
import matplotlib.pyplot as plt

def main(perc, data_file):
    # Check correct percentage
    assert((perc>=0) and (perc<100)) 
    # Read input point cloud and compute distance matrix 
    points = np.genfromtxt(data_file)
    # Take "perc" indices of samples and store into file 
    random.seed(0)
    f_ind_sampl = "output\\indices_sample.out"
    f_dist_X = "output\\dist_X.out"
    f_dist_S = "output\\dist_S.out"
    indices_sample = random.sample(range(points.shape[0]), int(points.shape[0]*perc))
    points_sample = points[indices_sample]
    np.savetxt(f_ind_sampl, indices_sample, fmt="%d", newline="\n")
    # Compute distance matrix of X and store 
    Dist_X = dist.squareform(dist.pdist(points))
    np.savetxt(f_dist_X, Dist_X, fmt="%.18e", delimiter=" ", newline="\n")
    # Compute distance matrix of sample and store 
    Dist_S = dist.squareform(dist.pdist(points[indices_sample]))
    np.savetxt(f_dist_S, Dist_S, fmt="%.18e", delimiter=" ", newline="\n")
    # Compute the matching and block funciton by calling "IBloFunMatch.exe" 
    os.system("x64\\Debug\\IBloFunMatchCPP.exe " + f_dist_S + " " + f_dist_X + " " + f_ind_sampl + " -d 2")
    ##################################################
    # Read data from output folder 
    #################################################
    # Read cycle representatives
    S_reps = [];
    with open("output/S_reps.out") as file:
        for line in file:
            rep_list = line.split(" ")[:-1]
            S_reps.append(list(np.array(rep_list).astype("int")))
        # end reading reps
    # end reading file
    S_reps_im = [];
    with open("output/S_reps_im.out") as file:
        for line in file:
            rep_list = line.split(" ")[:-1]
            S_reps_im.append(list(np.array(rep_list).astype("int")))
        # end reading reps
    # end reading file
    X_reps = [];
    with open("output/X_reps.out") as file:
        for line in file:
            rep_list = line.split(" ")[:-1]
            X_reps.append(list(np.array(rep_list).astype("int")))
        # end reading reps
    # end reading file
    pm_matrix = [];
    with open("output/pm_matrix.out") as file:
        for line in file:
            col_list = line.split(" ")[:-1]
            pm_matrix.append(list(np.array(col_list).astype("int")))
        # end reading reps
    # end reading file
    ##################################################
    # Generate cycle plots 
    #################################################
    fig, ax = plt.subplots(ncols=3, nrows=len(S_reps), figsize=(15,5*len(S_reps)))
    for idx_cycle, cycle_S in enumerate(S_reps):
        # CYCLE
        ax[idx_cycle, 0].scatter(points_sample[:,0], points_sample[:,1], zorder=2, c="navy")
        while (len(cycle_S)>0):
            edge = points[[cycle_S.pop(), cycle_S.pop()]]
            ax[idx_cycle, 0].plot(edge[:,0], edge[:,1], c="red", linewidth="5", zorder=1)
            ax[idx_cycle, 0].set_title(f"idx_cycle: {idx_cycle}")
        # end while
        # EMBEDDED CYCLE  
        ax[idx_cycle, 1].scatter(points[:,0], points[:,1], zorder=2, c="navy")
        cycle_im = S_reps_im[idx_cycle]
        while (len(cycle_im)>0):
            edge = points[[cycle_im.pop(), cycle_im.pop()]]
            ax[idx_cycle, 1].plot(edge[:,0], edge[:,1], c="red", linewidth="5", zorder=1)
        # end while
        # PIVOT CYCLE 
        ax[idx_cycle, 2].scatter(points[:,0], points[:,1], zorder=2, c="navy")
        if len(pm_matrix[idx_cycle])==0:
            continue
        # Plot pivot cycle 
        pivot_cycle = X_reps[pm_matrix[idx_cycle][-1]]
        while (len(pivot_cycle)>0):
            edge = points[[pivot_cycle.pop(), pivot_cycle.pop()]]
            ax[idx_cycle, 2].plot(edge[:,0], edge[:,1], c="red", linewidth="5", zorder=1)
            ax[idx_cycle, 2].set_title(f"Idx: {pm_matrix[idx_cycle][-1]}")
        # end while
    # end for
    plt.savefig("plots/cycles_image.png")
if __name__== "__main__":
    perc = float(sys.argv[1])
    data_file = sys.argv[2]
    main(perc, data_file)
