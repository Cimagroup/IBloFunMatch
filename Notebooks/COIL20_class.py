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
    classes_coil = [0,1] # list(range(2)) # [0, 4, 5]
    num_exp = 10
    # num_exp = 20
    # num_class = 20
    num_samples = 72
    data_percent = 0.4
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
        # end for 
        data_experiments.append(data_indices)
        scores_experiments_sum_class.append(sum_scores_experiment)
    # stop putting together indices of experiments 
    print("scores_experiments_sum_class")
    for id in range(num_exp):
        print(scores_experiments_sum_class[id])

    #################### Each CLASS into DATA #########################################################
    matching_class_total_partial = partial(matching_class_total, data, y)
    with Pool(processes=8) as pool:
        result_match_strength_class = pool.map(matching_class_total_partial, classes_coil)
    
    id_exp = 0
    with open("plots/COIL_CLASS/class_total_matchings.txt", mode="w") as file:
        for result in result_match_strength_class:
            strengths_exp = result[0]
            S_barcodes_1 = result[1]["S_barcode_1"]
            max_strengths_exp = np.max(strengths_exp)
            sum_strenghts_exp = np.sum(strengths_exp/S_barcodes_1[:,0])
            sq_strenghts = np.sqrt(np.sum(np.array(strengths_exp)**2))
            print(f"id class: {id_exp}, max: {max_strengths_exp:.2f}, sum: {sum_strenghts_exp:.2f}, sq: {sq_strenghts:.2f}", file=file)
            id_exp+=1

    barcode_X = result_match_strength_class[0][1]["X_barcode_1"]
    block_images = []
    for result in result_match_strength_class:
        ibmf_out = result[1]
        block_images += ibmf_out["block_function_1"]

    block_images_op = [i for i in range(len(barcode_X)) if i not in block_images]
    print("X_barcodes_1 not in image")
    print(barcode_X[block_images_op])


    #################### SUBSET to DATA #########################################################
    # In parallel, compute matchings from all sample data in experiments to total
    matching_experiment_total_partial = partial(matching_experiment_total, data, data_experiments, block_images_op)
    with Pool(processes=8) as pool:
        result_match_strength_exp = pool.map(matching_experiment_total_partial, range(num_exp))
    
    scores_mixed_bars = [] 
    X_barcode_1 = result[1]["X_barcode_1"]
    for result in result_match_strength_exp:
        block_function = result[1]["block_function_1"]
        S_barcode_1 = result[1]["S_barcode_1"]
        strengths_1 = result[1]["matching_strengths_1"]
        mix_score_sum = 0
        for idx, S_bar in enumerate(S_barcode_1): 
            if block_function[idx] in block_images_op:
                X_bar = X_barcode_1[block_function[idx]]
                # mix_score_sum += ((S_bar[1]-X_bar[0])*(X_bar[1]-S_bar[0])/(S_bar[0]*X_bar[0]))
                mix_score_sum += (S_bar[1]-X_bar[0]) * strengths_1[idx]
        # end for 
        scores_mixed_bars.append(mix_score_sum)
    #end for
        
    sum_strengths_list = []
    for idx, result in enumerate(result_match_strength_exp):
        sum_strengths_list.append(np.sum(result[0])-scores_mixed_bars[idx])
    print(np.array(sum_strengths_list))
    # order_indices = np.argsort(-np.array(sum_strengths_list)).tolist()
    order_indices = list(range(len(sum_strengths_list)))
    print(f"order_indices:{order_indices}")
    print(f"len(result_match_strength_exp):{len(result_match_strength_exp)}")
    result_match_strength_exp = [result_match_strength_exp[i] for i in order_indices]
    data_experiments = [data_experiments[i] for i in order_indices]
    scores_mixed_bars = [scores_mixed_bars[i] for i in order_indices]
    for id, data_indices in enumerate(data_experiments):
        np.savetxt(f"data_COIL20/indices_{id}.txt", data_indices, fmt='%d')
    id_exp = 0
    with open("plots/COIL_CLASS/experiments_matchings.txt", mode="w") as file:
        for id_exp, result in enumerate(result_match_strength_exp):
            strengths_exp = result[0]
            if len(strengths_exp)>0:
                max_strengths_exp = np.max(strengths_exp)
                sum_strengths_exp = np.sum(strengths_exp)
                new_sum_scores = sum_strengths_exp - scores_experiments_sum_class[id_exp]
                # sq_strengths = np.sqrt(np.sum(np.array(strengths_exp)**2))
                mix_score = scores_mixed_bars[id_exp]
                print(f"id exp: {order_indices[id_exp]}, sum: {sum_strengths_exp:.2f}", file=file)
                id_exp+=1


    

# main()
            
def matching_class_total(data, y, cidx):
    output_dir = f"output/output_{cidx}"
    os.makedirs(output_dir, exist_ok=True)
    indices_subset = np.nonzero(np.array(y)==cidx)[0]
    print(f"indices_subset:{indices_subset}")
    ibfm_out=  ibfm.get_IBloFunMatch_subset(
        None, data, indices_subset, output_dir, num_it=4, points=True, max_rad=-1
        )

    # DIM 1
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    if(ibfm_out["S_barcode_1"].shape[0]>0):
        ibfm.plot_matching(ibfm_out, ax, fig, dim=1, frame_on=True, strengths=False, block_function=True)
    ax[0].set_title(f"Class {cidx}", fontsize=20)
    plt.savefig(f"plots/COIL_CLASS/matching_1_class_{cidx}.png")
    plt.close(fig)
    # DIM 0
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    if(ibfm_out["S_barcode_0"].shape[0]>0):
        ibfm.plot_matching(ibfm_out, ax, fig, dim=0, frame_on=True, strengths=False, block_function=True)
    ax[0].set_title(f"Class {cidx}", fontsize=20)
    plt.savefig(f"plots/COIL_CLASS/matching_0_class_{cidx}.png")
    plt.close(fig)
    # Plot 0 diag 
    plot_zero_diag(ibfm_out, f"plots/COIL_CLASS/class_diag_{cidx}.png")

    block_scores = [] 
    already_matched = []
    for S_bar, match_id in zip(ibfm_out["S_barcode_1"], ibfm_out["block_function_1"]):
        X_bar = ibfm_out["X_barcode_1"][match_id]
        if match_id != -1 and match_id not in already_matched:
            # block_scores.append((X_bar[1]-X_bar[0])*(X_bar[1]-S_bar[0])/(S_bar[1]-X_bar[0]))
            block_scores.append((X_bar[1] - S_bar[0]))
            # already_matched.append(match_id)
        else:
            block_scores.append(0)

    return block_scores, ibfm_out

def matching_experiment_total(data, data_experiments, block_images_op, id_exp):
    output_dir = f"output/output_{id_exp}"
    ibfm_out=  ibfm.get_IBloFunMatch_subset(
        None, data, data_experiments[id_exp], output_dir, num_it=4, points=True, max_rad=-1
        )
    

    # DIM 1
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    if(ibfm_out["S_barcode_1"].shape[0]>0):
        ibfm.plot_matching(ibfm_out, ax, fig, dim=1, frame_on=True, strengths=False, codomain_int=block_images_op, block_function=True)
    ax[0].set_title(f"Experiment {id_exp}", fontsize=20)
    plt.savefig(f"plots/COIL_CLASS/matching_1_{id_exp}.png")
    plt.close(fig)
    # DIM 0
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    if(ibfm_out["S_barcode_0"].shape[0]>0):
        ibfm.plot_matching(ibfm_out, ax, fig, dim=0, frame_on=True, strengths=False, block_function=True)
    ax[0].set_title(f"Experiment {id_exp}", fontsize=20)
    plt.savefig(f"plots/COIL_CLASS/matching_0_{id_exp}.png")
    plt.close(fig)
    # Plot 0 diag 
    plot_zero_diag(ibfm_out, f"plots/COIL_CLASS/exp_diag_{id_exp}.png")
    # Compute block scores
    block_scores = [] 
    already_matched = []
    for idx_S, S_bar in enumerate(ibfm_out["S_barcode_1"]):
        match_id = ibfm_out["block_function_1"][idx_S]
        if match_id != -1 and match_id not in already_matched:
            X_bar = ibfm_out["X_barcode_1"][match_id]
            # block_scores.append((X_bar[1]-X_bar[0])*(X_bar[1]-S_bar[0])/(S_bar[1]-X_bar[0]))
            block_scores.append((X_bar[1] - S_bar[0]))
            already_matched.append(match_id)
        else:
            # block_scores.append(S_bar[0]-S_bar[1])
            block_scores.append(0)
        
    # # See also the matching from the complement 
    # data_complement = [i for i in range(data.shape[0]) if i not in data_experiments[id_exp]]
    # ibfm_out_complement =  ibfm.get_IBloFunMatch_subset(
    #     None, data, data_complement, output_dir, num_it=4, points=True, max_rad=-1
    #     )
    # # DIM 1
    # fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    # if(ibfm_out["S_barcode_1"].shape[0]>0):
    #     ibfm.plot_matching(ibfm_out_complement, ax, fig, dim=1, frame_on=True, strengths=False, codomain_int=block_images_op, block_function=True)
    # ax[0].set_title(f"Experiment {id_exp} complement", fontsize=20)
    # plt.savefig(f"plots/COIL_CLASS/matching_1_complement_{id_exp}.png")
    # plt.close(fig)
    return block_scores, ibfm_out

def matching_experiment_class(data, y, num_exp, subset_size, cidx):
    print(f"matching_experiment_class: num_exp:{num_exp}, subset_size:{subset_size}, class id:{cidx}")
    output_dir = f"output/output_{cidx}"
    os.makedirs(output_dir, exist_ok=True)
    rng = default_rng(5*cidx)
    class_indices = np.nonzero(np.array(y)==cidx)[0].tolist()
    class_data = data[class_indices]
    n_samples = class_data.shape[0]
    # Take a very well chosen subset
    # subset_idx_good = np.linspace(0, n_samples, subset_size+1).astype(int)[:-1].tolist()
    # exp_indices = [subset_idx_good]
    exp_indices = []
    # Take 10 subsets randomly
    for i in range(num_exp):
        exp_indices.append(np.sort(rng.choice(range(n_samples), subset_size, replace=False)).tolist()) 
    exp_match = []
    # exp_match_complement = []
    for id_exp, indices_subset in enumerate(exp_indices):
        indices_subset = exp_indices[id_exp]
        print(indices_subset)
        print(f"indices_subset length:{len(indices_subset)}, subset_size:{subset_size}")
        assert(len(indices_subset)==subset_size)
        exp_match.append(ibfm.get_IBloFunMatch_subset(None, class_data, indices_subset, output_dir, num_it=4, points=True, max_rad=-1))
        # also do matchings with complements
        # indices_complement = [i for i in range(class_data.shape[0]) if i not in indices_subset]
        # exp_match_complement.append(ibfm.get_IBloFunMatch_subset(
        #     None, class_data, indices_complement, output_dir, num_it=4, points=True, max_rad=-1
        # ))
    # end for 
    # Compute max and sum scores
    max_scores, sum_scores, sq_scores, st_scores = [], [], [], []
    scores_exp = []
    sums_zero_match = np.zeros(len(exp_match))
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
        already_matched = []
        for idx_S, S_bar in enumerate(ibfm_out["S_barcode_1"]):
            match_id = ibfm_out["block_function_1"][idx_S]
            if match_id != -1 and match_id not in already_matched:
                X_bar = ibfm_out["X_barcode_1"][match_id]
                # strength = ibfm_out["matching_strengths_1"][idx_S]
                # bar_affinity = (X_bar[1]-S_bar[0]) / (S_bar[1]-X_bar[0])
                # block_scores.append((X_bar[1]-S_bar[0]) * (S_bar[1]-S_bar[0]) / (X_bar[1]-X_bar[0]))
                # overlap_length = min(X_bar[1]-S_bar[0], complement_lengths[match_id])
                # block_scores.append((X_bar[1]-X_bar[0])*(X_bar[1]-S_bar[0])/(S_bar[1]-X_bar[0]))
                block_scores.append((X_bar[1] - S_bar[0]))
                already_matched.append(match_id)
            else:
                # block_scores.append(-(S_bar[1]-S_bar[0])/S_bar[0])
                block_scores.append(0)
        # end for 
        scores_exp.append(np.array(block_scores))
        
        # Get sum of intervals on zero diagram 
        for S_idx, S_bar in enumerate(ibfm_out["S_barcode_0"]):
            match_id = ibfm_out["block_function_0"][S_idx]
            if match_id != -1:
                X_bar = ibfm_out["X_barcode_0"][match_id]
                sums_zero_match[id_exp] += S_bar[1] - X_bar[0]
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
    print(f"class: {cidx}, order_indices:{order_indices}")
    # order_indices = list(range(len(st_scores)))
    print(f"order_indices:{order_indices}")
    max_scores = np.array(max_scores)[order_indices]
    sum_scores = np.array(sum_scores)[order_indices]
    sq_scores = np.array(sq_scores)[order_indices]
    sums_zero_match = np.array(sums_zero_match)[order_indices]
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
    # end plotting experiments per class
    # np.savetxt(f"plots/COIL_CLASS/class_{cidx}/sum_0_block.txt", sums_zero_match, fmt='%4.4f')
    # Write scores for this class and all experiments in a file 
    scores_matrix = np.vstack((np.argsort(sums_zero_match), np.argsort(-max_scores), np.argsort(-sum_scores), np.argsort(-sq_scores)))
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