import json
import numpy as np

from scipy.optimize import linear_sum_assignment
import numpy as np

def evaluate_retrieval_results(gt_tracks, results):
    recall_5 = 0
    recall_10 = 0
    mrr = 0
    count = 0
    for qid, query in enumerate(gt_tracks):
#         count += 1
#         if count < 92:
#             continue
        result = results[query]
        target = gt_tracks[query]
        try:
            rank = result.index(target)
#             if rank != 0:
#                 print(count)
        except ValueError:
            rank = 100
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(gt_tracks)
    recall_10 /= len(gt_tracks)
    mrr /= len(gt_tracks)
    print("Recall@5 is %.4f" % recall_5)
    print("Recall@10 is %.4f" % recall_10)
    print("MRR is %.4f" % mrr)

    
def best_match(sim_matrix):
    best_match = 1
    best_match_list = []
    if best_match==1:
        count = 0
        for i in range(len(sim_matrix)):
        #     print(i, np.argmax(sim_matrix[i,:]))
            row_best_idx = np.argmax(sim_matrix[i,:])
            col_best_idx = np.argmax(sim_matrix[:,row_best_idx])
            if col_best_idx == i:
                i_argsort = np.argsort(sim_matrix[i,:], axis=0)[::-1]
                top12 = sim_matrix[i,i_argsort[0]] - sim_matrix[i,i_argsort[1]]
                if top12 > 0:
                    temp = sim_matrix[row_best_idx, col_best_idx]
        #             sim_matrix[row_best_idx, :] -= 10
                    sim_matrix[:, col_best_idx] -= 10
                    best_match_list.append(i)
    #                 print(i, row_best_idx, col_best_idx)
                    count += 1
                    sim_matrix[row_best_idx, col_best_idx] = temp
        for i in range(len(sim_matrix)):
            if i not in best_match_list:
                row_best_idx = np.argmax(sim_matrix[i,:])
                if row_best_idx in best_match_list:
                    sim_matrix[i, row_best_idx] -= 10
    return sim_matrix

# if __name__ == '__main__':
#     with open("data/test-gt.json") as f:
#         gt_tracks = json.load(f)
#     with open("baseline/results.json") as f:
#         results = json.load(f)
#     evaluate_retrieval_results(gt_tracks, results)

json_save ="val_data_test_max_id.json"
with open(json_save, 'r') as f:  
    gt = json.load(f)

sim_matrix_path = "250sim_matrix.npy"
# sim_matrix_path = "130sim_matrix.npy"
sim_matrix_path = "./matrix/vrm.npy"
with open(sim_matrix_path, 'rb') as f:
    sim_matrix = np.load(f)

with open(sim_matrix_path, 'rb') as f:
    total_sim_matrix = np.load(f)
total_sim_matrix[total_sim_matrix<0]=0
sim_matrix = total_sim_matrix[:, 0, :]
sim_matrix += total_sim_matrix[:, 1, :]
sim_matrix += total_sim_matrix[:, 2, :]
sim_matrix /= 3
# other_sim_matrix = np.zeros((184, 184))
# for i in range(3, 46):
#     other_sim_matrix += total_sim_matrix[:, i, :]

# count_sim_matrix = np.zeros((184, 184))
# for r in range(184):
#     for c in range(184):
#         for i in range(3, 46):
#             if total_sim_matrix[r, i, c] != 0:
#                 count_sim_matrix[r, c] += 1
#         if count_sim_matrix[r, c] == 0:
#             count_sim_matrix[r, c] = 1
# other_sim_matrix /= count_sim_matrix
                

count_no_zero = np.zeros((184, 184))
count_no_zero = sim_matrix.copy()
# for row in range(0, 110):
#     for col in range(0, 110):
#         count_no_zero[row, col] = max(total_sim_matrix[row,:,col])

        

# for row in range(110, 184):
#     for col in range(110, 184):
#         count_no_zero[row, col] = max(total_sim_matrix[row,:,col])

            
# # other_sim_matrix /= count_no_zero
# # other_sim_matrix += count_no_zero
# # sim_matrix += 0.1 * other_sim_matrix
# # sim_matrix = np.zeros((184, 184))
# # sim_matrix += 1 * count_no_zero
sim_matrix += 1 * sim_matrix
# # sim_matrix += 1 * other_sim_matrix
# sim_matrix += 0.1 * other_sim_matrix
sim_matrix /= 2

color_path = '/dfs/data/others/clip/Tip-Adapter-main/pred_c1.npy'
color_path = './matrix/vcm.npy'
# color_path = '/dfs/data/others/clip/main/id_color_matrix.npy'
with open(color_path, 'rb') as f:
    id_color_matrix = np.load(f)
# print(id_color_matrix)
sim_matrix += 20*id_color_matrix
# type_path = '/dfs/data/others/clip/Tip-Adapter-main/pred_t15.npy'
# type_path = './matrix/vtm.npy'
# # type_path = '/dfs/data/others/clip/main/id_type_matrix.npy'
# with open(type_path, 'rb') as f:
#     id_type_matrix = np.load(f)
# sim_matrix += 20*id_type_matrix
# motion_path = '/dfs/data/others/clip/main/id_direct_matrix_v2.npy'
# motion_path = './matrix/vmm.npy'
# with open(motion_path, 'rb') as f:
#     id_direct_matrix = np.load(f)
# sim_matrix += 20*id_direct_matrix

# motion_path = './matrix/vsm2.npy'
# with open('/dfs/data/others/clip/main/id_nb_type_matrix.npy', 'rb') as f:
#     id_nb_type_matrix = np.load(f)

# # sim_matrix += 20*id_nb_type_matrix

# with open(motion_path, 'rb') as f:
#     id_nb_color_matrix = np.load(f)

# sim_matrix += 20*id_nb_color_matrix
# motion_path2 = '/dfs/data/others/aicity23/xclip/nb.npy'
# motion_path2 = './matrix/vsm1.npy'
# with open(motion_path2, 'rb') as f:
#     id_nb_matrix = np.load(f)

# sim_matrix += 1*id_nb_matrix

sim_matrix = best_match(sim_matrix)

order = sim_matrix.argsort(axis=1)
results = {}
# id_list = list(gt.keys())

text_video = {}
for key, value in gt.items():
    text_video[value['text_id']] = value['video_id']

for idx, line in enumerate(order):
    tid = gt[str(idx)]['text_id']
    results[tid] = []
    for jdx in line[::-1]:
        results[tid].append(gt[str(jdx)]['video_id'])
        
# with open("result7_8823k.json", "r") as f:
#     results = json.load(f)
evaluate_retrieval_results(text_video, results)  

with open("result19.json", "w") as w:
    json.dump(results, w)
    