import os
import numpy as np
import matplotlib as mpl

# Read executable path from cmake-generated file 
with open("../exe_path.txt") as f:
    EXECUTABLE_PATH = f.readline()

print(f"EXECUTABLE_PATH: {EXECUTABLE_PATH}")

attributes = ["X_barcode", "S_barcode", "X_reps", "S_reps", "S_reps_im", "pm_matrix", "induced_matching", "matching_strengths"]
types_list = ["float", "float", "int", "int", "int", "int", "int", "float"]

def get_IBloFunMatch_subset(Dist_S, Dist_X, idS, output_dir, max_rad=-1, num_it=1, store_0_pm=False):
    # Buffer files to write subsets and classes for communicating with C++ program 
    f_ind_sampl = output_dir + "/" + "indices_sample.out"
    f_dist_X = output_dir + "/" + "dist_X.out"
    f_dist_S = output_dir + "/" + "dist_S.out"
    output_data = {}
    # Compute distance matrices and save
    np.savetxt(f_ind_sampl, idS, fmt="%d", delimiter=" ", newline="\n")
    np.savetxt(f_dist_X, Dist_X, fmt="%.14e", delimiter=" ", newline="\n")
    np.savetxt(f_dist_S, Dist_S, fmt="%.14e", delimiter=" ", newline="\n")
    # Call IBloFunMatch C++ program (only for dimension 1 PH)
    extra_flags = ""
    if max_rad!=-1:
        extra_flags += " -r " + f"{max_rad:f}" + " "
    # added maximum radius flag
    if(num_it>1):
        extra_flags += " -i " + f"{num_it:d}" + " "
    # added number of collapses iteration flag
    if(store_0_pm):
        extra_flags += " -z true "
    # only if we want to store the 0 dimensional pm matrix

    os.system(EXECUTABLE_PATH + " " + f_dist_S + " " + f_dist_X + " " + f_ind_sampl + " -d 2 " + extra_flags )
    # Save barcodes and representatives reading them from output files
    data_read = []
    for dim in range(2):
        for attribute, typename in zip(attributes, types_list):
            if (attribute=="pm_matrix") and (dim==0) and (not store_0_pm):
                continue
            attribute_name = attribute + "_" + str(dim)
            with open(output_dir + "/" + attribute_name + ".out") as file:
                for line in file:
                    if(attribute=="matching_strengths"):
                        data_line = line.split(" ")[:-1]
                        data_read = list(np.array(data_line).astype(typename))
                        break
                    elif(attribute == "induced_matching"):
                        data_read.append(int(line))
                    else:
                        data_line = line.split(" ")
                        if (typename=="int"): # lines end with additional space
                            data_line=data_line[:-1]
                        data_read.append(list(np.array(data_line).astype(typename)))
                    # end if else 
                # end reading file lines 
                if typename=="float":
                    output_data[attribute_name] = np.array(data_read)
                else:
                    output_data[attribute_name] = data_read.copy()
                # end if-else 
            # end opening file 
            data_read.clear()
        # end saving all attributes 
        # end for 
    # Range over dimensions 0 and 1
    return output_data
# def get_IBloFunMatch_subset

def plot_matching(IBloFunMatch_o, output_dir, ax, fig, max_rad=-1, colorbars=["orange", "aquamarine"], frame_on=False, print_matching=False, dim=1):
    X_barcode = IBloFunMatch_o[f"X_barcode_{dim}"]
    S_barcode = IBloFunMatch_o[f"S_barcode_{dim}"]
    X_reps = IBloFunMatch_o[f"X_reps_{dim}"]
    induced_matching = IBloFunMatch_o[f"induced_matching_{dim}"]
    matching_strengths = IBloFunMatch_o[f"matching_strengths_{dim}"]
    if len(ax)!=2:
        raise ValueError

    lw_S, lw_X = 100/len(S_barcode), 100/len(X_barcode)
    for idx, bar in enumerate(S_barcode):
        # ax[0].plot([bar[0], bar[1]], [idx, idx], c="orange", linewidth=lw_S, zorder=1)
        ax[0].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[0], zorder=1))
    for idx, bar in enumerate(X_barcode):
        # ax[1].plot([bar[0], bar[1]], [idx, idx], c="aquamarine", linewidth=lw_X, zorder=1)
        ax[1].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[1], zorder=1))

    for ax_it in ax:
        ax_it.set_frame_on(frame_on)
        ax_it.set_yticks([])

    # Limits barcode diagrams on y axis
    ax[0].set_ylim([-1, S_barcode.shape[0]])
    ax[1].set_ylim([-1, X_barcode.shape[0]])
    # Limits on x axis depend on value where filtration is "cut"
    MAX_PLOT_RAD = np.max(X_barcode)*1.1
    if max_rad>=0:
        MAX_PLOT_RAD = max_rad
    ax[0].set_xlim([0, MAX_PLOT_RAD])
    ax[1].set_xlim([0, MAX_PLOT_RAD])

    # Plot Partial Matching
    for idx, idx_match in enumerate(induced_matching):
        if idx_match==-1:
            continue
        S_bar = S_barcode[idx]
        X_bar = X_barcode[idx_match]
        strength = matching_strengths[idx]
        if X_bar[1]<S_bar[0]:
            continue
        if print_matching:
            print(f"{S_bar} <--> {X_bar}, strength: {strength:.3f}")
        # Highlight matched bar sections 
        # ax[0].plot([S_bar[0], X_bar[1]], [idx, idx], c="navy", linewidth=lw_S, zorder=2, alpha=0.5)
        ax[0].add_patch(mpl.patches.Rectangle([S_bar[0], idx-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        # ax[1].plot([S_bar[0], X_bar[1]], [idx_match, idx_match], c="navy", linewidth=lw_X, zorder=2, alpha=0.5)
        ax[1].add_patch(mpl.patches.Rectangle([S_bar[0], idx_match-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        # Plot matchings
        pt_S = [S_bar[1], idx]
        pt_X = [X_bar[0], idx_match]
        con = mpl.patches.ConnectionPatch(
            xyA=pt_S, coordsA=ax[0].transData, 
            xyB=pt_X, coordsB=ax[1].transData,
            arrowstyle="-", connectionstyle='arc',
            color="blue", linewidth=2, zorder=4, 
            alpha = (strength/X_bar[1])
        )
        fig.add_artist(con)
    # end for
# end  def plot_matchi

def sampled_circle(r, R, n, RandGen):
    assert r<=R
    radii = RandGen.uniform(r,R,n)
    angles = RandGen.uniform(0,2*np.pi,n)
    return np.vstack((np.cos(angles)*np.sqrt(radii), np.sin(angles)*np.sqrt(radii))).transpose()
