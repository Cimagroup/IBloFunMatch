import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import distance

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition

from PIL import Image
import networkx as nx

import gudhi

import os

import IBloFunMatch_inter as ibfm

from multiprocessing import Pool
from functools import partial

def main():
    #################################################
    # Read COIL-20 dataset
    #################################################
    # classes_coil = list(range(20))
    classes_coil = [0,2] # list(range(2)) # [0, 4, 5]
    num_exp = 10
    # num_exp = 20
    # num_class = 20
    num_samples = 72
    # data_percent = 0.35
    data_percent = 0.5
    subset_size = int(num_samples*data_percent)
    print(f"subset_size:{subset_size}")
    y = []
    for c in classes_coil:
        y += [c]*num_samples
    data = []
    for c in classes_coil:
        for i in range(num_samples):
            im_frame = Image.open(f"coil-20-proc/coil-20-proc/obj{c+1}__{i}.png")
            np_frame = np.array(im_frame)
            data.append(np_frame.ravel())
        # samples per class
    # going through classes
    data = np.array(data)
    # Save into files
    os.makedirs(f"data_COIL20", exist_ok=True)
    os.makedirs(f"plots/COIL_CLASS/", exist_ok=True)
    np.savetxt(f"data_COIL20/y.txt", y, fmt='%d')
    # np.savetxt(f"data_COIL20/data.txt", data, fmt='%4.4f') TOO LARGE
    #################################################
    # Take samples of COIL-20 and perform matchings in parallel
    #################################################
    # Create directories for storing experiments
    for c in classes_coil:
        os.makedirs(f"plots/COIL_CLASS/class_{c}", exist_ok=True)

    
    matching_exp_partial = partial(matching_experiment_class, data, y, num_exp, subset_size)
    # for cidx in range(2):
    #     matching_exp_partial(cidx)
    with Pool(processes=8) as pool:
        result = pool.map(matching_exp_partial, classes_coil)

    # Indices of experiment into file
    data_experiments = []
    scores_experiments_sum_class = []
    for id in range(num_exp):
        data_indices = []
        sum_scores_experiment = 0
        for j in range(len(classes_coil)):
            assert(len(result[j][0][id])==subset_size)
            data_indices += result[j][0][id]
            sum_scores_experiment += result[j][1][id]
        # np.savetxt(f"data_COIL20/indices_{id}.txt", data_indices, fmt='%d')
        data_experiments.append(data_indices)
        scores_experiments_sum_class.append(sum_scores_experiment)
    # stop putting together indices of experiments 
    print("scores_experiments_sum_class")
    for id in range(num_exp):
        print(scores_experiments_sum_class[id])
    

# main()
            

def matching_experiment_class(data, y, num_bootstrap, subset_size, id_exp, cidx):
    output_dir = f"output/output_{id_exp}"
    os.makedirs(output_dir, exist_ok=True)
    rng = default_rng(5*cidx)
    class_indices = np.nonzero(np.array(y)==cidx)[0].tolist()
    class_data = data[class_indices]
    n_samples = class_data.shape[0]
    # Take a random subset
    sample_experiment = np.sort(rng.choice(range(n_samples), subset_size, replace=False)).tolist()
    # Take additional num_bootstrap samples 
    bootstrap_sample = []
    for i in range(num_bootstrap):
        bootstrap_sample.append(np.sort(rng.choice(range(n_samples), subset_size, replace=False)).tolist()) 
    bootstrap_match = []
    # Compute matchings over all bootstrap samples
    for id_exp, sample_bootstrap in enumerate(bootstrap_sample):
        union_indices = np.sort(np.unique(np.hstack((sample_bootstrap, sample_experiment))))
        union_data = class_data[union_indices]
        indices_sample = [i for i, idx in enumerate(union_indices) if idx in sample_experiment]
        indices_bootstrap = [i for i, idx in enumerate(union_indices) if idx in sample_bootstrap]
        match_sample = ibfm.get_IBloFunMatch_subset(None, union_data, indices_sample, output_dir, num_it=4, points=True, max_rad=-1)
        match_sample = ibfm.get_IBloFunMatch_subset(None, union_data, indices_sample, output_dir, num_it=4, points=True, max_rad=-1)
        
    # end for 
    # Compute max and sum scores
    max_scores, sum_scores, sq_scores, st_scores = [], [], [], []
    scores_exp = []
    for id_exp, ibfm_out in enumerate(exp_match):
        # Compute complement lengths
        # ibfm_compl = exp_match_complement[id_exp]
        # complement_lengths = np.zeros(ibfm_compl["X_barcode_1"].shape[0])
        # already_matched = []
        # for idx_S, S_bar in enumerate(ibfm_compl["S_barcode_1"]):
        #     match_id = ibfm_compl["block_function_1"][idx_S]
        #     if match_id != -1 and match_id not in already_matched:
        #         X_bar = ibfm_compl["X_barcode_1"][match_id]
        #         complement_lengths[match_id] = X_bar[1] - S_bar[0]
        #         already_matched.append(match_id)
        #     # end adding length
        # # end for over matched intervals in complement
        # Now, compute matching strength of current sample, taking into account 
        # the complement_lengths
        block_scores = [] 
        st_scores_exp = [0]
        already_matched = []
        for idx_S, S_bar in enumerate(ibfm_out["S_barcode_1"]):
            match_id = ibfm_out["block_function_1"][idx_S]
            if match_id != -1 and match_id not in already_matched:
                X_bar = ibfm_out["X_barcode_1"][match_id]
                # strength = ibfm_out["matching_strengths_1"][idx_S]
                block_scores.append((X_bar[1]-S_bar[0])*(S_bar[1]*X_bar[0]))
                # overlap_length = min(X_bar[1]-S_bar[0], complement_lengths[match_id])
                # block_scores.append((S_bar[1] - X_bar[0])*overlap_length)
                already_matched.append(match_id)
            else:
                block_scores.append(0)
        scores_exp.append(np.array(block_scores))
    #finished computing block scores 
    for id_exp, block_scores in enumerate(scores_exp):
        if(len(block_scores)>0):
            max_scores.append(np.max(block_scores))
            sum_scores.append(np.sum(block_scores))
            sq_scores.append(np.sqrt(np.sum(block_scores**2)))
        else:
            max_scores.append(0)
            sum_scores.append(0)
            sq_scores.append(0)
    # end adding scores
    # Sort according to scores 
    # order_indices = [0] + (np.argsort(-np.array(max_scores[1:]))+1).tolist()
    order_indices = np.argsort(-np.array(sum_scores)).tolist()
    # order_indices = list(range(len(st_scores)))
    print(f"order_simplices:{order_indices}")
    max_scores = np.array(max_scores)[order_indices]
    sum_scores = np.array(sum_scores)[order_indices]
    sq_scores = np.array(sq_scores)[order_indices]
    print(f"cidx:{cidx}")
    print(f"max_scores: {max_scores}")
    print(f"sum_scores: {sum_scores}")
    print(f"sq_scores: {sq_scores}")
    exp_match = [exp_match[i] for i in order_indices]
    # exp_match_complement = [exp_match_complement[i] for i in order_indices]
    exp_indices = [exp_indices[i] for i in order_indices]
    # Write max and sum scores
    # np.savetxt(f"plots/COIL_CLASS/class_{cidx}/max_scores.txt", max_scores, fmt='%4.4f')
    # np.savetxt(f"plots/COIL_CLASS/class_{cidx}/sum_scores.txt", sum_scores, fmt='%4.4f')
    # np.savetxt(f"plots/COIL_CLASS/class_{cidx}/sq_scores.txt", sq_scores, fmt='%4.4f')
    # Plot matchings in max order
    sums_zero_match = np.zeros(len(exp_match))
    for id_exp, ibfm_out in enumerate(exp_match):
        # PLOT DIM 1 
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
        if(ibfm_out["S_barcode_1"].shape[0]>0):
            ibfm.plot_matching(ibfm_out, ax, fig, dim=1, frame_on=True, strengths=False, block_function=True)
        ax[0].set_title(f"Experiment {id_exp}", fontsize=20)
        plt.savefig(f"plots/COIL_CLASS/class_{cidx}/matching_1_{id_exp}.png")
        plt.close(fig)
        # PLOT DIM 0
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
        if(ibfm_out["S_barcode_0"].shape[0]>0):
            ibfm.plot_matching(ibfm_out, ax, fig, dim=0, frame_on=True, strengths=False, block_function=True)
        ax[0].set_title(f"Experiment {id_exp}", fontsize=20)
        plt.savefig(f"plots/COIL_CLASS/class_{cidx}/matching_0_{id_exp}.png")
        plt.close(fig)
        # Plot 0 diag 
        plot_zero_diag(ibfm_out, f"plots/COIL_CLASS/class_{cidx}/diag_0_{id_exp}.png")
        # Get sum of intervals on zero diagram 
        for S_idx, S_bar in enumerate(ibfm_out["S_barcode_0"]):
            match_id = ibfm_out["block_function_0"][S_idx]
            X_bar = ibfm_out["X_barcode_0"][match_id]
            sums_zero_match[id_exp] += S_bar[1] - X_bar[0]
    # end plotting experiments per class
    # np.savetxt(f"plots/COIL_CLASS/class_{cidx}/sum_0_block.txt", sums_zero_match, fmt='%4.4f')
    # Write scores for this class and all experiments in a file 
    scores_matrix = np.vstack((np.argsort(sums_zero_match), np.argsort(max_scores), np.argsort(sum_scores), np.argsort(sq_scores)))
    scores_matrix = np.vstack((list(range(scores_matrix.shape[1])), scores_matrix))
    scores_matrix = scores_matrix.transpose()
    np.savetxt(f"plots/COIL_CLASS/class_{cidx}/scores_orders.txt", scores_matrix, fmt="%d")
    # Compute indices of experiment within all data 
    exp_indices_global = [] 
    for indices_subset in exp_indices: 
        aux_indices = np.array(class_indices)[indices_subset].tolist()
        assert(np.all(np.array(y)[aux_indices]==cidx)) # little check
        exp_indices_global.append(aux_indices)

    # # Plot complements of matchings 
    # for id_exp, ibfm_out_complement in enumerate(exp_match_complement):
    #     # PLOT DIM 1 
    #     fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    #     if(ibfm_out_complement["S_barcode_1"].shape[0]>0):
    #         ibfm.plot_matching(ibfm_out_complement, ax, fig, dim=1, frame_on=True, strengths=False, block_function=True)
    #     ax[0].set_title(f"Experiment {id_exp} complement", fontsize=20)
    #     plt.savefig(f"plots/COIL_CLASS/class_{cidx}/matching_1_{id_exp}_complement.png")
    #     plt.close(fig)
    # # getting exp_Indices
    return exp_indices_global, sum_scores
# matching_experiment_class

def plot_zero_diag(ibfm_out, filename):
    zero_barcode = [] 
    for bar, match in zip(ibfm_out["S_barcode_0"], ibfm_out["block_function_0"]):
        if match>-1:
            zero_barcode.append([ibfm_out["X_barcode_0"][match][1], bar[1]])

    if len(zero_barcode)>0:  
        barcode = np.array(zero_barcode)
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4,4))
        ax = gudhi.plot_persistence_diagram(persistence=barcode, axes=ax)
        ax.set_title(f"Entropy:{entropy(barcode):.3f}")
        plt.savefig(filename)
        plt.close(fig)


def entropy(barcode):
    differences = barcode[:,1]-barcode[:,0]
    differences = differences[differences>0]
    total_length = np.sum(differences)
    coefficients = differences/total_length
    entropy = 0
    for coeff in coefficients:
        entropy -= coeff * np.log(coeff)

    return entropy 

if __name__=="__main__":
    main()