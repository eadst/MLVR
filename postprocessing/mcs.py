import json
import numpy as np


def best_match(sim_matrix):
    best_match = 1
    best_match_list = []
    if best_match==1:
        count = 0
        for i in range(len(sim_matrix)):
            row_best_idx = np.argmax(sim_matrix[i,:])
            col_best_idx = np.argmax(sim_matrix[:,row_best_idx])
            if col_best_idx == i:
                i_argsort = np.argsort(sim_matrix[i,:], axis=0)[::-1]
                top12 = sim_matrix[i,i_argsort[0]] - sim_matrix[i,i_argsort[1]]
                if top12 > 0:
                    temp = sim_matrix[row_best_idx, col_best_idx]
                    sim_matrix[:, col_best_idx] -= 10
                    best_match_list.append(i)
                    count += 1
                    sim_matrix[row_best_idx, col_best_idx] = temp
        for i in range(len(sim_matrix)):
            if i not in best_match_list:
                row_best_idx = np.argmax(sim_matrix[i,:])
                if row_best_idx in best_match_list:
                    sim_matrix[i, row_best_idx] -= 10
    return sim_matrix


def merge_matrix(path_dict):
    
    # video recognition module
    sim_matrix_path = path_dict["baseline"]
    with open(sim_matrix_path, 'rb') as f:
        total_sim_matrix = np.load(f)
    total_sim_matrix[total_sim_matrix<0]=0
    sim_matrix = total_sim_matrix[:, 0, :]
    sim_matrix += total_sim_matrix[:, 1, :]
    sim_matrix += total_sim_matrix[:, 2, :]
    sim_matrix /= 3

    count_no_zero = np.zeros((184, 184))
    count_no_zero = sim_matrix.copy()

    sim_matrix += 1 * sim_matrix
    sim_matrix /= 2

    # vehicle color module
    color_path = path_dict["vcm"]
    with open(color_path, 'rb') as f:
        color_matrix = np.load(f)
    sim_matrix += 20*color_matrix
    
    # vehicle type module
    type_path = path_dict["vtm"]
    with open(type_path, 'rb') as f:
        type_matrix = np.load(f)
    sim_matrix += 20*type_matrix

    # vehicle motion module
    motion_path = path_dict["vmm"]
    with open(motion_path, 'rb') as f:
        motion_matrix = np.load(f)
    sim_matrix += 20*motion_matrix

    # vehicle surrounding module branch 1
    vsm1_path = path_dict["vsm1"]
    with open(vsm1_path, 'rb') as f:
        vsm1_matrix = np.load(f)
    sim_matrix += 1*vsm1_matrix
    
    # vehicle surrounding module branch 2
    vsm2_path = path_dict["vsm2"]
    with open(vsm2_path, 'rb') as f:
        vsm2_matrix = np.load(f)
    sim_matrix += 20*vsm2_matrix

    # match control system
    sim_matrix = best_match(sim_matrix)
        
    # generate text-video json result
    # load text id
    queries_path = "/dfs/data/others/aicity23/data/test-queries.json"
    with open(queries_path) as f:
        queries_json = json.load(f) 
    text_ids = list(queries_json.keys())
    # load video id
    tracks_path = "/dfs/data/others/aicity23/data/test-tracks.json"
    with open(tracks_path) as f:
        tracks_json = json.load(f)  
    video_ids = list(tracks_json.keys())
    
    # sort the score in each row
    order = sim_matrix.argsort(axis=1)
    results = {}
    for idx, line in enumerate(order):
        tid = text_ids[idx]
        results[tid] = []
        for jdx in line[::-1]:
            results[tid].append(video_ids[jdx])

    # save result json for testing
    with open("results.json", "w") as w:
        json.dump(results, w)
    
    
path_dict = {"baseline": "./matrix/vrm.npy", # video recognition module
             "vcm": "./matrix/vcm.npy", # vehicle color module
             "vtm": "./matrix/vtm.npy", # vehicle type module
             "vmm": "./matrix/vmm.npy", # vehicle motion module
             "vsm1": "./matrix/vsm1.npy", # vehicle surrounding module branch 1
             "vsm2": "./matrix/vsm2.npy", # vehicle surrounding module branch 2
            }
merge_matrix(path_dict)