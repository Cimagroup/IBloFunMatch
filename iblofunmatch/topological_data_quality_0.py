import numpy as np
import gudhi
import matplotlib.pyplot as plt
import matplotlib as mpl

import scipy.spatial.distance as dist
import itertools
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np

def read_csr_matrix(cs_matrix):
    """Function to read output from minimum_spanning_tree and prepare it as a list of 
    filtration values (in order) together with an array with the corresponding pairs.
    """ 
    filtration_list = []
    pairs = []
    entry_idx = 0
    for i, cummul_num_entries in enumerate(cs_matrix.indptr[1:]):
        while entry_idx < cummul_num_entries:
            pairs.append((i, cs_matrix.indices[entry_idx]))
            filtration_list.append(cs_matrix.data[entry_idx])
            entry_idx+=1
    # Sort filtration values and pairs
    pairs_arr = np.array(pairs)
    np.argsort(filtration_list)
    sort_idx = np.argsort(filtration_list)
    filtration_list = np.array(filtration_list)[sort_idx].tolist()
    pairs_arr = pairs_arr[sort_idx]
    return filtration_list, pairs_arr

def filtration_pairs(points):
    """Returns the persistent homology pairs and filtration values for 
    a given point sample. This is like a 0-dimensional persistent homology 
    wrapper for the minimum_spanning_tree function.
    """ 
    mst = minimum_spanning_tree(dist.squareform(dist.pdist(points)))
    # We now read the compressed sparse row matrix
    filtration_list, pairs_arr = read_csr_matrix(mst)
    # Get proper merge tree pairs 
    labels = np.array(list(range(points.shape[0])))
    correct_pairs_list = []
    for pair in pairs_arr:
        min_label = np.min(labels[pair])
        max_label = np.max(labels[pair])
        correct_pairs_list.append([min_label, max_label])
        assert min_label < max_label
        labels[labels==max_label]=min_label
    # end updating correct pairs
    pairs_arr = np.array(correct_pairs_list)
    return filtration_list, pairs_arr


def add_columns_mod_2(col1, col2):
    """ Given two lists of integers, which are sparse representations of a pair of vectors in Z mod 2, this funciton adds them and 
    returns the result in the same input format.
    """
    diff_1 = set(col1).difference(set(col2))
    diff_2 = set(col2).difference(set(col1))
    result = diff_1.union(diff_2)
    return list(result)


def get_inclusion_matrix(pairs_arr_S, pairs_arr_X, subset_indices):
    """ Given two pairs of arrays with the vertex merge pairs, this function returns the associated inclusion matrix. 
    From the point of view of minimum spanning trees, the output matrix columns can be interpreted as the minimum paths that are needed to 
    go through in mst(X) in order to connect the endpoints from an edge in mst(S)
    """
    pivot2column = [-1] + np.argsort(pairs_arr_X[:,1]).tolist()
    inclusion_matrix = []
    for col_S in pairs_arr_S:
        col_S = [subset_indices[i] for i in col_S]
        col_M = []
        while(len(col_S)>0):
            piv = np.max(col_S)
            col_M.append(pivot2column[piv])
            col_S = add_columns_mod_2(col_S, pairs_arr_X[pivot2column[piv]])
        # end reducing column S
        col_M.sort()
        inclusion_matrix.append(col_M)
    return inclusion_matrix

def get_inclusion_matrix_pivots(matrix_list, num_rows):
    """ Returns the pivots of a matrix given in list format"""
    pivots = []
    pivot2column = np.ones(num_rows, dtype="int")*-1
    for i, column in enumerate(matrix_list):
        reduce_column = list(column)
        piv = np.max(reduce_column)
        while(pivot2column[piv]>-1):
            reduce_column = add_columns_mod_2(reduce_column, matrix_list[pivot2column[piv]])
            piv = np.max(reduce_column)
            # we assume that columns are never reduced to the 0 column
        pivots.append(piv)
        pivot2column[piv] = i
    # end getting pivots
    return pivots  


def plot_matching_0(filt_S, filt_X, matching, ax):
    """ Given two zero dimensional barcodes as well as a block function between them, this function plots the associated diagram"""
    # Plot matching barcode
    for i, X_end in enumerate(filt_X):
        if i in matching:
            S_end = filt_S[matching.index(i)]
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], X_end, 0.4, color="navy", zorder=2))
            ax.add_patch(mpl.patches.Rectangle([X_end*0.9, i-0.2], S_end-X_end, 0.4, color="orange", zorder=1.9))
        else:
            ax.add_patch(mpl.patches.Rectangle([0, i-0.2], X_end, 0.4, color="aquamarine", zorder=2))

    MAX_PLOT_RAD = max(np.max(filt_S), np.max(filt_X))*1.1
    ax.set_xlim([-0.1*MAX_PLOT_RAD, MAX_PLOT_RAD*1.1])
    ax.set_ylim([-0.1*len(filt_S), len(filt_X)])
    ax.set_frame_on(False)
    ax.set_yticks([])
#end plot_matching_0

def plot_density_matrix(filt_S, filt_X, matching, ax, nbins=5):
    endpoints_X = np.array(filt_X)[matching]
    differences = np.array([[a, a-b] for (a,b) in zip(filt_S, endpoints_X)])
    ax.hist2d(differences[:,0], differences[:,1], bins=(nbins, nbins), cmap=plt.cm.jet)
    ax.set_xlabel('Differences of the bars')
    ax.set_ylabel('Length of the bars X')

def plot_density_matrix_percentage(filt_S, filt_X, matching, ax, nbins=5):
    ends_S = np.array(filt_S)
    max_end = np.max(ends_S)
    ends_diff = ends_S - np.array(filt_X)[matching]
    Diag_diff = np.vstack((ends_S, ends_diff)).transpose()
    hist = np.histogram2d(Diag_diff[:,1], Diag_diff[:,0], bins=nbins, range=[[0,max_end], [0,max_end]])[0]
    sum_cols = np.sum(hist, axis=0)
    sum_cols = np.maximum(sum_cols, 1)
    hist = np.divide(hist, sum_cols)
    hist=hist[-1::-1]
    ax.imshow(hist, extent=(0, max_end, 0, max_end))
    ax.set_xlabel('Differences of the bars (percent)')
    ax.set_ylabel('Length of the bars X')

def compute_mathing_diagram(filt_S, filt_X, matching, _tol=1e-5):
    pairs = []
    for i, a in enumerate(filt_S):
        b = filt_X[matching[i]]
        pairs.append((a, b))
    # end for 
    multiplicities = [] 
    pairs_copy = list(pairs)
    old_pair = pairs_copy.pop()
    multiplicities.append(1)
    # First compute matched pairs and their multiplicities
    while len(pairs_copy)>0:
        pair = pairs_copy.pop()
        if np.all(np.abs(np.array(pair)-np.array(old_pair)) < _tol):
            multiplicities[-1]+=1
        else:
            multiplicities.append(1)
            old_pair = pair
        # end checking if pair similar
    # end while
    # Next compute right infinity points and their multiplicities 
    unmatched_idx = [j for j in range(len(filt_X)) if j not in matching]
    if len(unmatched_idx)>0:
        b = filt_X[unmatched_idx.pop()]
        pairs.append((np.infty, b))
        multiplicities.append(1)
        while len(unmatched_idx)>0:
            prev_b = b
            b = filt_X[unmatched_idx.pop()]
            if (np.abs(b-prev_b)<_tol):
                multiplicities[-1]+=1
            else:
                pairs.append((np.inf, b))
                multiplicities.append(1)
            # end checking same endpoint
        # iterate over unmatched intervals
    # only if some intervals are unmatched
    return np.array(pairs), multiplicities


def plot_matching_diagram(pairs, ax, colorpt="black", marker="o", size=15, hmax=None):
    fin_pairs = pairs[pairs[:,0]<np.inf]
    if hmax is None:
        max_x = max(np.max(fin_pairs), np.max(pairs[:,1]))
        lim_x = max_x*1.3
        infty_x = max_x*1.1
    else:
        max_x, lim_x, infty_x = hmax, hmax*1.3, hmax*1.1
    # end if-else loop
    ax.plot([0, lim_x], [0, lim_x], c="gray", linewidth=1, zorder=1)
    ax.plot([infty_x, infty_x], [0, lim_x], c="blue", linewidth=1, zorder=1)
    ax.scatter(fin_pairs[:,0], fin_pairs[:,1], color=colorpt, s=size, marker=marker, zorder=2)
    inf_points = pairs[pairs[:,0]==np.inf]
    ax.scatter(np.ones(len(inf_points))*infty_x, inf_points[:,1], color=colorpt, s=size, marker=marker, zorder=2)
    ax.text(infty_x*1.02, infty_x*1.1, "∞", fontsize=15, color="blue")
    ax.scatter([infty_x], [infty_x], color=colorpt, s=size, marker=marker, zorder=2)
    # Adjust spines 
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom","left"]].set_position("zero")
    # adjust margins 
    ax.set_xlim([0,lim_x])
    ax.set_ylim([0,lim_x])
