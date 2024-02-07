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

import os

import IBloFunMatch_inter as ibfm

from multiprocessing import Pool
from functools import partial

def main():
    #################################################
    # Read COIL-20 dataset
    #################################################
    num_class = 20
    num_samples = 72
    y = []
    for c in range(num_class):
        y += [c]*num_samples
    data = []
    for c in range(1, num_class+1):
        for i in range(num_samples):
            im_frame = Image.open(f"coil-20-proc/coil-20-proc/obj{c}__{i}.png")
            np_frame = np.array(im_frame)
            data.append(np_frame.ravel())
        # samples per class
    # going through classes
    data = np.array(data)
    # Save into files
    os.makedirs(f"data_COIL20", exist_ok=True)
    np.savetxt(f"data_COIL20/y.txt", y, fmt='%d')
    np.savetxt(f"data_COIL20/data.txt", data, fmt='%4.4f')
    #################################################
    # Take samples of COIL-20 and perform matchings in parallel
    #################################################
    # Create directories for storing experiments
    for i in range(20):
        os.makedirs(f"plots/COIL_CLASS/class_{i}", exist_ok=True)

    # run dataset in parallel 
    data_percent = 0.3
    # for cidx in range(2):
    #     class_indices = np.nonzero(np.array(y)==cidx)[0].tolist()
    #     matching_experiment_class(data, y, data_percent, cidx)
    #end over classes 
    num_exp = 2
    matching_exp_partial = partial(matching_experiment_class, data, y, num_exp, data_percent)
    # for cidx in range(2):
    #     matching_exp_partial(cidx)
    num_classes = 8
    with Pool(processes=8) as pool:
        result = pool.map(matching_exp_partial, range(num_classes))

    # Indices of experiment into file
    for id in range(num_exp):
        data_indices = []
        for j in range(num_classes):
            data_indices += result[j][id]
        np.savetxt(f"data_COIL20/indices_{id}.txt", data_indices, fmt='%4.4f')
        raise(ValueError)
    
# main()

def matching_experiment_class(data, y, num_exp, data_percent, cidx):
    output_dir = f"output/output_{cidx}"
    os.makedirs(output_dir, exist_ok=True)
    rng = default_rng(cidx)
    class_indices = np.nonzero(np.array(y)==cidx)[0].tolist()
    class_data = data[class_indices]
    n_samples = class_data.shape[0]
    # Take a very well chosen subset
    # subset_idx_good = np.linspace(0, n_samples, int(n_samples*data_percent)).astype(int)[:-1].tolist()
    # exp_indices = [subset_idx_good]
    exp_indices = []
    # Take 10 subsets randomly
    for i in range(num_exp):
        exp_indices.append(np.sort(rng.choice(range(n_samples), int(n_samples*data_percent), replace=False)).tolist()) 
    exp_match = []
    for id_exp in range(len(exp_indices)):
        # indices_subset = [class_indices.index(i) for i in exp_indices[id_exp]]
        # exp_indices.append(indices_subset) 
        indices_subset = exp_indices[id_exp]
        exp_match.append(ibfm.get_IBloFunMatch_subset(None, class_data, indices_subset, output_dir, num_it=4, points=True, max_rad=-1))
    # finished with matchings
    # Compute max and sum scores
    max_scores, sum_scores = [], []
    for id_exp, ibfm_out in enumerate(exp_match):
        if(ibfm_out["S_barcode_1"].shape[0]>0):
            max_scores.append(np.max(ibfm_out["matching_strengths_1"]))
            sum_scores.append(np.sum(ibfm_out["matching_strengths_1"]))
        else:
            max_scores.append(0)
            sum_scores.append(0)
    # end adding scores
    # Sort according to scores 
    order_indices = np.argsort(-np.array(max_scores))
    max_scores = np.array(max_scores)[order_indices]
    sum_scores = np.array(sum_scores)[order_indices]
    exp_match = [exp_match[i] for i in order_indices]
    # Write max and sum scores
    np.savetxt(f"plots/COIL_CLASS/class_{cidx}/max_scores.txt", max_scores, fmt='%4.4f')
    np.savetxt(f"plots/COIL_CLASS/class_{cidx}/sum_scores.txt", sum_scores, fmt='%4.4f')
    # Plot matchings in max order
    for id_exp, ibfm_out in enumerate(exp_match):
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
        if(ibfm_out["S_barcode_1"].shape[0]>0):
            ibfm.plot_matching(ibfm_out, output_dir, ax, fig, dim=1, frame_on=True, strengths=False)
        plt.savefig(f"plots/COIL_CLASS/class_{cidx}/matching_1_{id_exp}.png")
    # Return sample class datasets 
    # data_exp = []
    # for id_exp in order_indices:
    #     data_exp.append(class_data[exp_indices[id_exp]])

    print(exp_indices[0])
    print(order_indices)
    print("====")
    exp_indices_sorted = [] 
    for idx in order_indices:
        exp_indices_sorted.append(np.array(class_indices)[exp_indices[idx]].tolist())
    print(exp_indices_sorted)
    return exp_indices_sorted
# matching_experiment_class

if __name__=="__main__":
    main()