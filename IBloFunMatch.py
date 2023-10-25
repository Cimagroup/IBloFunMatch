# Functions for computing and plotting the induced block function and matchings 
from operator import iand
from re import S, X
import sys 
import os
import numpy as np
import scipy.spatial.distance as dist
import random
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from gudhi import plot_persistence_diagram

def main(exe_file, perc, data_file):
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
    os.system(exe_file + " " + f_dist_S + " " + f_dist_X + " " + f_ind_sampl + " -d 2")
    ##################################################
    # Read data from output folder 
    #################################################
    # Read barcodes 
    X_barcode = []
    with open("output/X_barcode.out") as file:
        for line in file:
            rep_list = line.split(" ")
            X_barcode.append(np.array(rep_list).astype("float").ravel())
        # end reading bar
    # end reading file
    X_barcode = np.array(X_barcode)
    print("X_barcode")
    print(X_barcode)
    S_barcode = []
    with open("output/S_barcode.out") as file:
        for line in file:
            rep_list = line.split(" ")
            S_barcode.append(list(np.array(rep_list).astype("float")))
        # end reading bar
    # end reading file
    S_barcode = np.array(S_barcode)
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
    # Read matrix and matching
    with open("output/pm_matrix.out") as file:
        for line in file:
            col_list = line.split(" ")[:-1]
            pm_matrix.append(list(np.array(col_list).astype("int")))
        # end reading reps
    # end reading file
    induced_matching = [];
    with open("output/induced_matching.out") as file:
        for line in file:
            induced_matching.append(int(line))
        # end reading reps
    # end reading file
    print("Induced Matching")
    print(induced_matching)
    ##################################################
    # Generate cycle plots illustrating induced matching
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
        if induced_matching[idx_cycle]==-1:
            continue
        # Plot pivot cycle 
        pivot_cycle = X_reps[induced_matching[idx_cycle]]
        while (len(pivot_cycle)>0):
            edge = points[[pivot_cycle.pop(), pivot_cycle.pop()]]
            ax[idx_cycle, 2].plot(edge[:,0], edge[:,1], c="red", linewidth="5", zorder=1)
            ax[idx_cycle, 2].set_title(f"Idx: {induced_matching[idx_cycle]}")
        # end while
    # end for
    plt.savefig("plots/cycles_image.png")
    ##################################################
    # Persistence Diagrams with Induced Matching
    #################################################
    # Plot induced matching between two diagrams
    max_val = max(np.max(S_barcode), np.max(X_barcode))
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(15,7.5))
    # S_barcode_ext = np.vstack((S_barcode, np.array([[0,max_val]])))
    S_barcode_ext = [(1, bar) for bar in S_barcode]+[(2,[0,max_val])]
    X_barcode_ext = [(1, bar) for bar in X_barcode]+[(2,[0,max_val])]
    plot_persistence_diagram(S_barcode_ext, axes=ax[0], colormap=["black", "black", "white"])
    plot_persistence_diagram(X_barcode_ext, axes=ax[1], colormap=["black", "black", "white"])
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    ax[0].set_title("1 PH sample")
    ax[1].set_title("1 PH dataset")
    for idx, idx_match in enumerate(induced_matching):
        pt_S = S_barcode[idx]
        if idx_match==-1:
            continue
        pt_X = X_barcode[idx_match]
        con = ConnectionPatch(
            xyA=pt_S, coordsA=ax[0].transData, 
            xyB=pt_X, coordsB=ax[1].transData,
            arrowstyle="-", connectionstyle='arc',
            color="red", linewidth=3
        )
        fig.add_artist(con)
    # end for 
    plt.savefig("plots/ind_matching_PD.png")
    # Plot induced matching within the same diagram
# end main 
    

if __name__== "__main__":
    exe_file = sys.argv[1]
    perc = float(sys.argv[2])
    data_file = sys.argv[3]
    main(exe_file, perc, data_file)
