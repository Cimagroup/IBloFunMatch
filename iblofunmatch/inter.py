import os
import numpy as np
import matplotlib as mpl
from pathlib import Path

# Read executable path from cmake-generated file 
parent_dir = Path(os.path.realpath(__file__)).parent.parent
exe_path = os.path.join(parent_dir, "exe_path_Debug.txt")
with open(exe_path) as f:
    EXECUTABLE_PATH = f.readline()

print(f"EXECUTABLE_PATH: {EXECUTABLE_PATH}")

attributes = ["X_barcode", "S_barcode", "X_reps", "S_reps", "S_reps_im", "pm_matrix", "induced_matching", "matching_strengths", "block_function"]
types_list = ["float", "float", "int", "int", "int", "int", "int", "float", "int"]

def get_IBloFunMatch_subset(Dist_S, Dist_X, idS, output_dir, max_rad=-1, num_it=1, store_0_pm=False, points=False, max_dim=2):
    # Buffer files to write subsets and classes for communicating with C++ program 
    f_ind_sampl = output_dir + "/" + "indices_sample.out"
    f_dist_X = output_dir + "/" + "dist_X.out"
    f_dist_S = output_dir + "/" + "dist_S.out"
    output_data = {}
    # Compute distance matrices and save
    np.savetxt(f_ind_sampl, idS, fmt="%d", delimiter=" ", newline="\n")
    if points:
        with open(f_dist_X, "w") as f:
            f.write("OFF\n")
            f.write(f"{Dist_X.shape[0]} 0 0\n")
            np.savetxt(f, Dist_X, fmt="%.14e", delimiter=" ", newline="\n")
    else:
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
    if(points):
        extra_flags += " -c true "
    # only if we want to store the 0 dimensional pm matrix

    os.system(EXECUTABLE_PATH + f" {f_dist_S} {f_dist_X} {f_ind_sampl} -d {max_dim} -o {output_dir}" + extra_flags )
    # Save barcodes and representatives reading them from output files
    data_read = []
    for dim in range(2):
        if max_dim < dim:
            break
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
                    elif(attribute == "block_function"):
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

def plot_matching(IBloFunMatch_o, ax, fig, max_rad=-1, colorbars=["orange", "aquamarine", "red", "yellow"], frame_on=False, print_matching=False, dim=1, strengths=True, block_function=False, codomain_int=[], repeated_codomain=[]):
    if block_function:
        strengths=False
    X_barcode = IBloFunMatch_o[f"X_barcode_{dim}"]
    S_barcode = IBloFunMatch_o[f"S_barcode_{dim}"]
    X_reps = IBloFunMatch_o[f"X_reps_{dim}"]
    if block_function:
        matching = IBloFunMatch_o[f"block_function_{dim}"]
    else:
        matching = IBloFunMatch_o[f"induced_matching_{dim}"]
    
    matching_strengths = IBloFunMatch_o[f"matching_strengths_{dim}"]
        
    if len(ax)!=2:
        print(f"ERROR: len(ax) should be 2 but it is {len(ax)}")
        raise ValueError

    lw_S, lw_X = 100/len(S_barcode), 100/len(X_barcode)
    for idx, bar in enumerate(S_barcode):
        ax[0].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[0], zorder=1))
    for idx, bar in enumerate(X_barcode):
        if idx in codomain_int:
            ax[1].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[2], zorder=1))
        elif idx in repeated_codomain:
            ax[1].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, alpha=0.5, color=colorbars[3], zorder=3))
        else:
            ax[1].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[1], zorder=1))

    for ax_it in ax:
        ax_it.set_frame_on(frame_on)
        ax_it.set_yticks([])

    # Limits barcode diagrams on y axis
    ax[0].set_ylim([-1, S_barcode.shape[0]])
    ax[1].set_ylim([-1, X_barcode.shape[0]])
    # Limits on x axis depend on value where filtration is "cut"
    MAX_PLOT_RAD = max(np.max(S_barcode), np.max(X_barcode))*1.1
    if max_rad>=0:
        MAX_PLOT_RAD = max(max_rad, MAX_PLOT_RAD)
    ax[0].set_xlim([0, MAX_PLOT_RAD])
    ax[1].set_xlim([0, MAX_PLOT_RAD])

    # Plot Partial Matching
    for idx, idx_match in enumerate(matching):
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
        ax[0].add_patch(mpl.patches.Rectangle([S_bar[0], idx-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        ax[1].add_patch(mpl.patches.Rectangle([S_bar[0], idx_match-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        # Plot matchings
        pt_S = [S_bar[1], idx]
        pt_X = [X_bar[0], idx_match]
        if strengths:
            alpha = strength/X_bar[1]
        else:
            alpha = 0.7
        con = mpl.patches.ConnectionPatch(
            xyA=pt_S, coordsA=ax[0].transData, 
            xyB=pt_X, coordsB=ax[1].transData,
            arrowstyle="-", connectionstyle='arc',
            color="blue", linewidth=2, zorder=4, 
            alpha = alpha
        )
        fig.add_artist(con)
    # end for
# end  def plot_matchi

def plot_from_block_function(S_barcode, X_barcode, block_function, fig, ax, max_rad=-1, colorbars=["orange", "aquamarine", "red"], frame_on=False):    
    if len(ax)!=2:
        print(f"ERROR: len(ax) should be 2 but it is {len(ax)}")
        raise ValueError

    lw_S, lw_X = 100/len(S_barcode), 100/len(X_barcode)
    for idx, bar in enumerate(S_barcode):
        ax[0].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[0], zorder=1))
    for idx, bar in enumerate(X_barcode):
        ax[1].add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=colorbars[1], zorder=1))

    for ax_it in ax:
        ax_it.set_frame_on(frame_on)
        ax_it.set_yticks([])

    # Limits barcode diagrams on y axis
    ax[0].set_ylim([-1, S_barcode.shape[0]])
    ax[1].set_ylim([-1, X_barcode.shape[0]])
    # Limits on x axis depend on value where filtration is "cut"
    MAX_PLOT_RAD = max(np.max(S_barcode), np.max(X_barcode))*1.1
    if max_rad>=0:
        MAX_PLOT_RAD = max(max_rad, MAX_PLOT_RAD)
    ax[0].set_xlim([0, MAX_PLOT_RAD])
    ax[1].set_xlim([0, MAX_PLOT_RAD])

    # Plot Partial Matching
    for idx, idx_match in enumerate(block_function):
        if idx_match==-1:
            continue
        S_bar = S_barcode[idx]
        X_bar = X_barcode[idx_match]
        if X_bar[1]<S_bar[0]:
            continue
        # Highlight matched bar sections 
        ax[0].add_patch(mpl.patches.Rectangle([S_bar[0], idx-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        ax[1].add_patch(mpl.patches.Rectangle([S_bar[0], idx_match-0.2], (X_bar[1]-S_bar[0]), 0.4, color="navy", zorder=2))
        # Plot matchings
        pt_S = [S_bar[1], idx]
        pt_X = [X_bar[0], idx_match]
        alpha = 0.7
        con = mpl.patches.ConnectionPatch(
            xyA=pt_S, coordsA=ax[0].transData, 
            xyB=pt_X, coordsB=ax[1].transData,
            arrowstyle="-", connectionstyle='arc',
            color="blue", linewidth=2, zorder=4, 
            alpha = alpha
        )
        fig.add_artist(con)
    # end for
# end  def plot_matchi

def plot_blofun_0_diag(ibfm_out, ax, draw_hist=False):
    """ Given two zero dimensional barcodes as well as a block function between them, this function plots the associated diagram"""
    S_barcode_0 = ibfm_out["S_barcode_0"]
    if draw_hist:
        weights = [-np.max(S_barcode_0)/2 / S_barcode_0.shape[0]]*S_barcode_0.shape[0]
        ax.hist(S_barcode_0[:,1], label='CDF', histtype='stepfilled', alpha=0.8, color=mpl.colormaps["RdBu"](0.15), weights=weights, cumulative=True)
    # end hist
    X_barcode_0 = ibfm_out["X_barcode_0"]
    block_0 = ibfm_out["block_function_0"]
    Diag_0 = []
    for S_bar, match_id in zip(S_barcode_0, block_0):
        X_bar = X_barcode_0[match_id]
        Diag_0.append([(S_bar[1] + X_bar[1])/2, (S_bar[1] - X_bar[1])/2])
    
    Diag_0 = np.array(Diag_0)
    ax.scatter(Diag_0[:,0], Diag_0[:,1], s=30, color=mpl.colormaps["RdBu"](1/1.1), zorder=1, marker="x")
    ax.plot([0,np.max(Diag_0)*1.1], [0,0], linewidth=3, color="gray", zorder=0)
    if draw_hist:
        ax.set_ylim([-np.max(S_barcode_0)*1.1/2, np.max(S_barcode_0)/2])
    else:
        ax.set_ylim([-0.05*np.max(S_barcode_0), np.max(S_barcode_0)/2])
    ax.set_aspect("equal")
#end plot_blofun_0_diag
    
def plot_XYZ_matching_0(exp_ibfm, ax):
    # find matched and unmatched intervals 
    block_0_0 = exp_ibfm[0]["block_function_0"]
    block_0_1 = exp_ibfm[1]["block_function_0"]
    repeated_block_0 = [i for i in block_0_0 if i in block_0_1]
    # plot unmatched bars as points 
    Z_barcode_0 = exp_ibfm[0]["X_barcode_0"]
    X_barcode_0 = exp_ibfm[0]["S_barcode_0"]
    Y_barcode_0 = exp_ibfm[1]["S_barcode_0"]
    # plot vertical line 
    ax.plot([0,0], [0, Z_barcode_0.shape[0]], c="gray", zorder=0)
    # plot profile of 0 barcode of Z in both sides
    ax.plot(Z_barcode_0[:,1], range(Z_barcode_0.shape[0]), c="gray")
    ax.plot(-Z_barcode_0[:,1], range(Z_barcode_0.shape[0]), c="gray")
    # Get block functions
    blofun_X_0 = exp_ibfm[0]["block_function_0"]
    blofun_Y_0 = exp_ibfm[1]["block_function_0"]
    
    for idx_Z, Z_bar_x in enumerate(Z_barcode_0[:,1]):
        # Base bar in gray
        ax.add_patch(mpl.patches.Rectangle([-Z_bar_x, idx_Z-0.2], 2*Z_bar_x, 0.4, color="gray", zorder=1, alpha=0.5))
        # Matched bar from X
        if idx_Z in blofun_X_0:
            X_bar_x = X_barcode_0[blofun_X_0.index(idx_Z)][1]
            ax.add_patch(mpl.patches.Rectangle([-X_bar_x, idx_Z-0.2], X_bar_x, 0.4, color="blue", zorder=1.5))
        # end if 
        # Matched bar from Y
        if idx_Z in blofun_Y_0:
            Y_bar_x = Y_barcode_0[blofun_Y_0.index(idx_Z)][1]
            ax.add_patch(mpl.patches.Rectangle([0, idx_Z-0.2], Y_bar_x, 0.4, color="red", zorder=1.5))
        # end if 
    # end for 
    ax.set_yticks([])

def plot_barcode(barcode, color, ax):
    lw = 100/len(barcode)
    for idx, bar in enumerate(barcode):
        ax.add_patch(mpl.patches.Rectangle([bar[0], idx-0.2], (bar[1]-bar[0]), 0.4, color=color, zorder=1))

    ax.set_yticks([])

    # Limits barcode diagrams on y axis
    ax.set_ylim([-1, barcode.shape[0]])
    # Limits on x axis depend on value where filtration is "cut"
    MAX_PLOT_RAD = np.max(barcode)*1.1
    
    ax.set_xlim([0, MAX_PLOT_RAD])
# end  plot barcode

def sampled_circle(r, R, n, RandGen):
    assert r<=R
    radii = RandGen.uniform(r,R,n)
    angles = RandGen.uniform(0,2*np.pi,n)
    return np.vstack((np.cos(angles)*radii, np.sin(angles)*radii)).transpose()
