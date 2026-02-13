import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.spatial.distance as dist
import itertools

# Read executable path from cmake-generated file 
parent_dir = Path(os.path.realpath(__file__)).parent.parent
exe_path = os.path.join(parent_dir, "exe_path_Debug.txt")
with open(exe_path) as f:
    EXECUTABLE_PATH = f.readline()

print(f"EXECUTABLE_PATH: {EXECUTABLE_PATH}")

attributes = ["X_barcode", "S_barcode", "X_reps", "S_reps", "S_reps_im", "pm_matrix", "induced_matching", "block_function"]
types_list = ["float", "float", "int", "int", "int", "int", "int", "int"]

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
    if(num_it<0):
        raise ValueError("Number of iterations parameter (num_it) needs to be >= 0")
    else:
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
                    if(attribute == "induced_matching"):
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

def plot_matching(IBloFunMatch_o, ax, fig, max_rad=-1, colorbars=["orange", "aquamarine", "red", "yellow"], frame_on=False, print_matching=False, dim=1, block_function=False, codomain_int=[], repeated_codomain=[]):
    if block_function:
        strengths=False
    X_barcode = IBloFunMatch_o[f"X_barcode_{dim}"]
    S_barcode = IBloFunMatch_o[f"S_barcode_{dim}"]
    X_reps = IBloFunMatch_o[f"X_reps_{dim}"]
    if block_function:
        matching = IBloFunMatch_o[f"block_function_{dim}"]
    else:
        matching = IBloFunMatch_o[f"induced_matching_{dim}"]
    
        
    if len(ax)!=2:
        print(f"ERROR: len(ax) should be 2 but it is {len(ax)}")
        raise ValueError
    
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
        if X_bar[1]<S_bar[0]:
            continue
        if print_matching:
            print(f"{S_bar} <--> {X_bar}")
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

def plot_from_block_function(S_barcode, X_barcode, block_function, fig, ax, max_rad=-1, colorbars=["orange", "aquamarine", "red"], frame_on=False):    
    if len(ax)!=2:
        print(f"ERROR: len(ax) should be 2 but it is {len(ax)}")
        raise ValueError

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

def plot_matching_0(ibfm_out, ax):
    """ Given two zero dimensional barcodes as well as a block function between them, this function plots the associated diagram"""
    S_barcode_0 = ibfm_out["S_barcode_0"]
    X_barcode_0 = ibfm_out["X_barcode_0"]
    block_function_0 = ibfm_out["block_function_0"]
    # Plot matching barcode
    for i, X_end in enumerate(X_barcode_0[:,1]):
        if i in block_function_0:
            S_end = S_barcode_0[:,1][block_function_0.index(i)]
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], X_end, 0.4, color="navy", zorder=2))
            ax.add_patch(mpl.patches.Rectangle([X_end*0.9, i-0.2], S_end-X_end, 0.4, color="orange", zorder=1.9))
        else:
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], X_end, 0.4, color="aquamarine", zorder=2))

    MAX_PLOT_RAD = max(np.max(S_barcode_0), np.max(X_barcode_0))*1.1
    ax.set_xlim([-0.1*MAX_PLOT_RAD, MAX_PLOT_RAD*1.1])
    ax.set_ylim([-0.1*X_barcode_0.shape[0], X_barcode_0.shape[0]])
    ax.set_frame_on(False)
    ax.set_yticks([])
#end plot_matching_0

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

### Geometric Matching 
def compute_components(edgelist, num_points):
    components = np.array(range(num_points))
    for edge in edgelist:
        max_idx = np.max(components[edge])
        min_idx = np.min(components[edge])
        indices = np.nonzero(components == components[max_idx])[0]
        components[indices]=np.ones(len(indices))*components[min_idx]
    
    return components

def plot_geometric_matching(a, b, idx_S, X, ibfm_out, ax, _tol=1e-5, labelsize=10):
    S = X[idx_S]
    # Obtain indices of bars that are approximately equal to a and b, these go from (a_idx - a_shift) to a_idx. (same for b_idx)
    a_idx = np.max(np.nonzero(ibfm_out["S_barcode_0"][:,1] < a + _tol))
    a_shift = np.sum(ibfm_out["S_barcode_0"][:,1][:a_idx+1] > a - _tol)
    b_idx = np.max(np.nonzero(ibfm_out["X_barcode_0"][:,1] < b + _tol))
    b_shift = np.sum(ibfm_out["X_barcode_0"][:,1][:b_idx+1] > b - _tol)
    pair_ab = [a_idx, b_idx]
    shift_ab = [a_shift, b_shift]
    num_points = X.shape[0]
    for idx in range(3):
        ax[idx].scatter(S[:,0], S[:,1], color=mpl.colormaps["RdBu"](0.3/1.3), s=60, marker="o", zorder=2)
        ax[idx].scatter(X[:,0], X[:,1], color=mpl.colormaps["RdBu"](1/1.3), s=40, marker="x", zorder=1)
        # Plot edges that came before a, b
        bool_smaller = dist.pdist(S)<=a-_tol
        edgelist = np.array([[i,j] for (i,j) in itertools.product(idx_S, idx_S) if i < j])[bool_smaller].tolist()
        bool_smaller = dist.pdist(X)<=b-_tol
        edgelist += np.array([[i,j] for (i,j) in itertools.product(range(num_points), range(num_points)) if i < j])[bool_smaller].tolist()
        for edge in edgelist:
            ax[idx].plot(X[edge][:,0], X[edge][:,1], c="black", zorder=0.5)
        # Remove axis 
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])
        # Draw node labels
        for i in range(X.shape[0]):
            ax[idx].text(X[i,0]+0.05, X[i,1], f"{i}", fontsize=labelsize)
        # end for labels 
    # end for plots
    # Plot edges from a 
    bool_smaller = dist.pdist(S)<a+_tol
    edgelist = np.array([[i,j] for (i,j) in itertools.product(idx_S, idx_S) if i < j])[bool_smaller].tolist()
    for edge in edgelist:
        ax[0].plot(X[edge][:,0], X[edge][:,1], c="black", zorder=0.5)
    # # Plot edges from b
    bool_smaller = dist.pdist(X) <b +_tol
    edgelist = np.array([[i,j] for (i,j) in itertools.product(range(num_points), range(num_points)) if i < j])[bool_smaller].tolist()
    for edge in edgelist:
        ax[2].plot(X[edge][:,0], X[edge][:,1], c="black", zorder=0.5)
    # Now, plot cycle graph of components 
    ax[3].set_xticks([])
    ax[3].set_yticks(list(range(num_points)))
    components_mat = []
    for idx in range(3):
        edgelist = ibfm_out['S_reps_0'][:pair_ab[0]-shift_ab[0]*int(idx!=0)+1]
        edgelist += ibfm_out['X_reps_0'][:pair_ab[1]-shift_ab[1]*int(idx!=2)+1]
        components = compute_components(edgelist, num_points)
        components_mat.append(components)
    
    components_mat = np.array(components_mat)
    for idx in range(3):
        u_components = np.unique(components_mat[idx]).tolist()
        points = np.array([np.ones(len(u_components))*idx, u_components]).transpose()
        ax[3].scatter(points[:,0], points[:,1], c="black", zorder=2)
        if idx==1:
            for comp in u_components:
                col_idx = components_mat[1].tolist().index(comp)
                left_comp = components_mat[0, col_idx]
                right_comp = components_mat[2, col_idx]
                ax[3].plot([0,1,2],[left_comp, comp, right_comp], c="black")
    
    
    # Adjust frames a bit more far appart
    for idx in range(4):
        xlim = ax[idx].get_xlim()
        xlength = xlim[1]-xlim[0]
        xlim = (xlim[0]-xlength*0.1, xlim[1]+xlength*0.1)
        ylim = ax[idx].get_ylim()
        ylength = ylim[1]-ylim[0]
        ylim = (ylim[0]-ylength*0.1, ylim[1]+ylength*0.1)
        ax[idx].set_xlim(xlim)
        ax[idx].set_ylim(ylim)
        if idx < 3:
            ax[idx].set_aspect("equal")
    
    # Write titles 
    ax[0].set_title(f"{a:.2f}+, {b:.2f}-")
    ax[1].set_title(f"{a:.2f}-, {b:.2f}-")
    ax[2].set_title(f"{a:.2f}-, {b:.2f}+")
    ax[3].set_title(f"G({a:.2f},{b:.2f})")

def plot_density_matrix(dim, exp_ibfm, nbins=5):
    bars_X=exp_ibfm['X_barcode_'+str(dim)]
    bars_S=exp_ibfm['S_barcode_'+str(dim)]
    block = exp_ibfm['block_function_'+str(dim)]
    matched_bars = [(bars_X[i],b) for (i,b) in zip(block,bars_S) if i!=-1]
    unmatched_bars = [b for (i,b) in zip(block,bars_S) if i==-1]
    diff_length = np.array([[sum(abs(x-y)),y[1]-y[0]] for (x,y) in matched_bars])
    diff_unmatched = np.array([[b[1]-b[0],np.inf] for b in unmatched_bars])
    if len(diff_unmatched)!=0:
        diff_length=np.concatenate((diff_length,diff_unmatched))
    plt.hist2d(diff_length[:,0], diff_length[:,1], bins=(nbins, nbins), cmap=plt.cm.jet)
    plt.xlabel('Differences of the bars')
    plt.ylabel('Length of the bars X')
    plt.colorbar()
    return

def plot_bimodule_0_blofun(ibfm_output, ax, fig, fill=True, linecolor="gray"):
    S_ends = ibfm_output["S_barcode_0"][:,1]
    X_ends = ibfm_output["X_barcode_0"][:,1]
    blofun = ibfm_output["block_function_0"]
    unmatched = [i for i in range(len(X_ends)) if i not in blofun]
    hlim = np.max(S_ends)*1.6
    # Fill horizontal turquoise bands for unmatched intervals 
    for idx in unmatched:
        b = X_ends[idx]
        if fill:
            ax.add_patch(mpl.patches.Rectangle([0, 0], hlim,  b, facecolor="turquoise", edgecolor="none", alpha=0.2, zorder=0.5))
        ax.plot([0,hlim], [b,b], linewidth=1, color=linecolor, zorder=1)
    # end for 
     # Fill matched intervals with corresponding orange rectangles
    match_points = []
    for idx, idx_match in enumerate(blofun):
        a = S_ends[idx]
        b = X_ends[idx_match]
        match_points.append([a,b])
        if fill:
            ax.add_patch(mpl.patches.Rectangle([0, 0], a, b, facecolor="orange", edgecolor="none", alpha=0.4, zorder=0.6))
        ax.plot([0,a,a], [b,b,0], linewidth=1, color=linecolor, zorder=1)
    # for 
    match_points = np.array(match_points)
    ax.scatter(match_points[:,0], match_points[:,1], s=30, marker="o", color=linecolor)
    # draw diagonal line for reference 
    ax.plot([0,hlim], [0, hlim], linestyle="dashed", linewidth=1, color="black", zorder=0.3)
    # Set limits 
    ax.set_xlim([0, hlim*0.9])
    ax.set_ylim([0, np.max(X_ends)*1.3])
    ax.set_aspect("equal")
    fig.tight_layout()


### Random Circle Creation 
def sampled_circle(r, R, n, RandGen):
    assert (r<=R) and (0 <= r)
    radii = RandGen.uniform(r,R,n)
    angles = RandGen.uniform(0,2*np.pi,n)
    return np.vstack((np.cos(angles)*radii, np.sin(angles)*radii)).transpose()

def circle(r, n):
    assert r > 0
    angles = np.linspace(0,2*np.pi,n)[:-1]
    return np.vstack((np.cos(angles)*r, np.sin(angles)*r)).transpose()



# Helper functions for plotting matchings, Vietoris-Rips complexes and their cycle representatives
# Some of the code for these comes from the TDQUAL repository: https://github.com/Cimagroup/tdqual/tree/main/tdqual

import gudhi

def plot_cycle(Z, cycle_edges, ax, color="red", linewidth="5", label=None, dashes=None):
    cycle_edges = list(cycle_edges)
    while (len(cycle_edges)>0):
        edge = Z[[cycle_edges.pop(), cycle_edges.pop()]]
        line, = ax.plot(edge[:,0], edge[:,1], c=color, linewidth=linewidth, zorder=1, label=label)
        if dashes!= None:
            line.set_dashes(dashes)
    # end while
# end def

def plot_Vietoris_Rips(Z, filt_val, ax, labels=False, fontsize=15, color="black"):
    # Plot point cloud
    ax.scatter(Z[:,0], Z[:,1], color=color, s=20, marker="o", zorder=1)
    # Plot simplicial complex 
    rips_complex = gudhi.RipsComplex(points=Z, max_edge_length=filt_val)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.collapse_edges()
    simplex_tree.expansion(2)
    edgelist = []
    triangles = []
    for filtered_value in simplex_tree.get_filtration():
        simplex = filtered_value[0]
        if len(simplex)==2:
            edgelist.append(simplex)
            ax.plot(Z[simplex][:,0], Z[simplex][:,1], linewidth=2, c=color, zorder=0.5)
        if len(simplex)==3:
            triangles.append(mpl.patches.Polygon(Z[simplex]))
        # end triangles
    # end for 
    triangles_collection = mpl.collections.PatchCollection(triangles, color=color, alpha=0.1, zorder=-1)
    ax.add_collection(triangles_collection)
    ax.set_aspect("equal")
    # Adjust margins
    xscale = ax.get_xlim()[1]-ax.get_xlim()[0]
    yscale = ax.get_ylim()[1]-ax.get_ylim()[0]
    xlim = ax.get_xlim()
    xlim = (xlim[0]-xscale*0.1, xlim[1]+xscale*0.1)
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()
    ylim = (ylim[0]-yscale*0.1, ylim[1]+yscale*0.1)
    ax.set_ylim(ylim)
    # Plot labels
    if labels:
        components = compute_components(edgelist, Z.shape[0])
        # Point Labels 
        for i in range(Z.shape[0]):
            ax.text(Z[i,0]-0.035*xscale, Z[i,1]-0.035*yscale, f"{components[i]}", fontsize=fontsize, color="white", fontweight="bold")

    # Finish with aspect details 
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.grid(True, color="gray", alpha=0.2)

def plot_Vietoris_Rips_subset(Z, X_indices, filt_val, ax, fontsize=15):
    X = Z[X_indices]
    # Plot point cloud
    ax.scatter(X[:,0], X[:,1], color=mpl.colormaps["RdBu"](0.3/1.3), s=60, marker="o", zorder=2, label="X")
    ax.scatter(Z[:,0], Z[:,1], color=mpl.colormaps["RdBu"](1/1.3), s=40, marker="o", zorder=1, label="$Z\setminus X$")
    # Plot simplicial complex 
    rips_complex = gudhi.RipsComplex(points=Z, max_edge_length=filt_val)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    simplex_tree.expansion(2)
    edgelist = []
    triangles_Z, triangles_X = [], []
    for filtered_value in simplex_tree.get_filtration():
        simplex = filtered_value[0]
        if len(simplex)==2:
            edgelist.append(simplex)
            if len(np.intersect1d(simplex, X_indices))==2:
                ax.plot(Z[simplex][:,0], Z[simplex][:,1], linewidth=2, c=mpl.colormaps["RdBu"](0.3/1.3), alpha=0.2, zorder=0.5)
            else:
                ax.plot(Z[simplex][:,0], Z[simplex][:,1], linewidth=2, c=mpl.colormaps["RdBu"](1/1.3), alpha=0.2, zorder=0.5)
        if len(simplex)==3:
            if len(np.intersect1d(simplex, X_indices))==3: 
                triangles_X.append(mpl.patches.Polygon(Z[simplex]))
            else:
                triangles_Z.append(mpl.patches.Polygon(Z[simplex]))
        # end triangles
    # end for 
    triangles_X_collection = mpl.collections.PatchCollection(triangles_X, color=mpl.colormaps["RdBu"](0.3/1.3), alpha=0.1, zorder=-1)
    triangles_Z_collection = mpl.collections.PatchCollection(triangles_Z, color=mpl.colormaps["RdBu"](1/1.3), alpha=0.1, zorder=-1.2)
    ax.add_collection(triangles_X_collection)
    ax.add_collection(triangles_Z_collection)
    ax.set_aspect("equal")
    # Adjust margins
    xscale = ax.get_xlim()[1]-ax.get_xlim()[0]
    yscale = ax.get_ylim()[1]-ax.get_ylim()[0]
    xlim = ax.get_xlim()
    xlim = (xlim[0]-xscale*0.1, xlim[1]+xscale*0.1)
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()
    ylim = (ylim[0]-yscale*0.1, ylim[1]+yscale*0.1)
    ax.set_ylim(ylim)

    # Finish with aspect details 
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.grid(True, color="gray", alpha=0.2)
