import os
import numpy as np
import matplotlib as mpl

EXECUTABLE_PATH = f"../build/IBloFunMatch" # this is my particular path
attributes = ["X_barcode", "S_barcode", "X_reps", "S_reps", "S_reps_im", "pm_matrix", "induced_matching", "matching_strengths"]
types_list = ["float", "float", "int", "int", "int", "int", "int", "float"]

def get_IBloFunMatch_subset(Dist_S, Dist_X, idS, output_dir):
    # Buffer files to write subsets and classes for communicating with C++ program 
    f_ind_sampl = output_dir + "indices_sample.out"
    f_dist_X = output_dir + "dist_X.out"
    f_dist_S = output_dir + "dist_S.out"
    output_data = {}
    # Compute distance matrices and save
    np.savetxt(f_ind_sampl, idS, fmt="%d", delimiter=" ", newline="\n")
    np.savetxt(f_dist_X, Dist_X, fmt="%.14e", delimiter=" ", newline="\n")
    np.savetxt(f_dist_S, Dist_S, fmt="%.14e", delimiter=" ", newline="\n")
    # Call IBloFunMatch C++ program (only for dimension 1 PH)
    os.system(EXECUTABLE_PATH + " " + f_dist_S + " " + f_dist_X + " " + f_ind_sampl + " -d 2")
    # Save barcodes and representatives reading them from output files
    data_read = []
    for attribute_name, typename in zip(attributes, types_list):
        print(f"attribute:{attribute_name}, type:{typename}")
        print(output_dir + attribute_name + ".out")
        with open(output_dir + attribute_name + ".out") as file:
            for line in file:
                if(attribute_name=="matching_strengths"):
                    data_line = line.split(" ")[:-1]
                    print(data_line)
                    data_read = list(np.array(data_line).astype(typename))
                    break
                elif(attribute_name == "induced_matching"):
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
    return output_data
# def get_IBloFunMatch_output_range

def plot_matching(IBloFunMatch_o, output_dir, ax, fig):
    X_barcode = IBloFunMatch_o["X_barcode"]
    S_barcode = IBloFunMatch_o["S_barcode"]
    X_reps = IBloFunMatch_o["X_reps"]
    induced_matching = IBloFunMatch_o["induced_matching"]
    if len(ax)!=2:
        raise ValueError

    lw_S, lw_X = 100/len(S_barcode), 100/len(X_barcode)
    for idx, bar in enumerate(S_barcode):
        # ax[0].add_patch(mpl.patches.Rectangle((bar[0], idx-(lw_S/2)), bar[1]-bar[0], lw_S, zorder=1, color="orange"))
        ax[0].plot([bar[0], bar[1]], [idx, idx], c="orange", linewidth=lw_S, zorder=1)
    for idx, bar in enumerate(X_barcode):
        # ax[1].add_patch(mpl.patches.Rectangle((bar[0], idx-(lw_X/2)), bar[1]-bar[0], lw_X, zorder=1,  color="aquamarine"))
        ax[1].plot([bar[0], bar[1]], [idx, idx], c="aquamarine", linewidth=lw_X, zorder=1)

    for ax_it in ax:
        ax_it.set_frame_on(False)
        ax_it.set_yticks([])

    # Plot Partial Matching
    for idx, idx_match in enumerate(induced_matching):
        if idx_match==-1:
            continue
        S_bar = S_barcode[idx]
        X_bar = X_barcode[idx_match]
        if X_bar[1]<S_bar[0]:
            continue
        print(f"{S_bar} <--> {X_bar}, len: {X_bar[1]-S_bar[0]:.3f}")
        # Highlight matched bar sections 
        # ax[0].add_patch(mpl.patches.Rectangle((S_bar[0], idx-(lw_S/2)), X_bar[1]-S_bar[0], lw_S, zorder=2, alpha=0.5, color="navy"))
        # ax[1].add_patch(mpl.patches.Rectangle((S_bar[0], idx_match-(lw_X/2)), X_bar[1]-S_bar[0], lw_X, zorder=2, alpha=0.5, color="navy"))
        ax[0].plot([S_bar[0], X_bar[1]], [idx, idx], c="navy", linewidth=lw_S, zorder=2, alpha=0.5)
        ax[1].plot([S_bar[0], X_bar[1]], [idx_match, idx_match], c="navy", linewidth=lw_X, zorder=2, alpha=0.5)
        # Plot matchings
        pt_S = [S_bar[1], idx]
        pt_X = [X_bar[0], idx_match]
        con = mpl.patches.ConnectionPatch(
            xyA=pt_S, coordsA=ax[0].transData, 
            xyB=pt_X, coordsB=ax[1].transData,
            arrowstyle="-", connectionstyle='arc',
            color="navy", linewidth=2, zorder=4, #alpha=0.3 #, alpha=match_prevalence
            alpha = (IBloFunMatch_o["matching_strengths"][idx]/max( IBloFunMatch_o["matching_strengths"]))
        )
        fig.add_artist(con)
    # end for
# end  def plot_matchi

def sampled_circle(num_pts, rad, margin_pcnt, rng):
    radius = rng.uniform(rad-margin_pcnt*rad, rad+margin_pcnt*rad, num_pts)
    angle = np.pi * rng.uniform(0, 2, num_pts)
    return np.vstack((np.sqrt(radius) * np.cos(angle), np.sqrt(radius) * np.sin(angle))).transpose()
# def sampled_circle 
