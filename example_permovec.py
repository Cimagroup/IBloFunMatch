import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

def main(ex_permovec, perc, filename):
    # Read points and indices of sample
    points = np.loadtxt(filename, skiprows=2)
    # Read points and indices of sample
    points = np.loadtxt(filename, skiprows=2)
    random.seed(0)
    assert((perc>=0) and (perc<100))
    indices_sample = random.sample(range(points.shape[0]), int(points.shape[0]*perc))
    # Save samples into file "indices_sample.out"
    with open("output\\indices_sample.out", "w") as file:
        for idx in indices_sample:
            file.write(str(idx)+"\n")
            
    os.system(ex_permovec + " -d 2 " + filename + " output\\indices_sample.out")
    points_sample = points[indices_sample]
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
    # Generate Plots
    fig, ax = plt.subplots(ncols=3, nrows=len(S_reps), figsize=(15,5*len(S_reps)))
    viridis = mpl.colormaps['viridis']
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
        if(len(pm_matrix[idx_cycle])==1):
            column_cycles = [X_reps[pm_matrix[idx_cycle][-1]]]
        else:
            column_cycles = [X_reps[entry] for entry in pm_matrix[idx_cycle]]
        for cycle_count, entry_cycle in enumerate(column_cycles):
            while (len(entry_cycle)>0):
                edge = points[[entry_cycle.pop(), entry_cycle.pop()]]
                ax[idx_cycle, 2].plot(edge[:,0], edge[:,1], c=viridis(cycle_count/(len(S_reps)+1)), linewidth="5", zorder=1)
                ax[idx_cycle, 2].set_title(f"Idx: {pm_matrix[idx_cycle][-1]}")
            # end while
        # end for 
    # end for
    plt.savefig("plots/cycles_image.png")
    
if __name__=="__main__":
    ex_permovec = sys.argv[1] # path to "ex_permovec" .exe file 
    perc = float(sys.argv[2]) # percentage of sample 
    filename=sys.argv[3] # path to point cloud file 
    main(ex_permovec, perc, filename)
