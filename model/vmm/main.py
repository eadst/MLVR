import json
import sys
from statistics import mode
from sklearn.linear_model import LinearRegression
import numpy as np


def motion_test_info(output_path):
    tracks_path = "/dfs/data/others/aicity23/data/test-tracks.json"
    with open(tracks_path) as f:
        tracks_json = json.load(f)  
    queries_path = "/dfs/data/others/aicity23/data/test-queries.json"
    with open(queries_path) as f:
        queries_json = json.load(f)  
    keywords = {}
    new_direct_lines = ""
    direction_dict = {"straight": 0, "forward":0, "down": 0, "through": 0, "stop": 1, "left": 2, "right": 3}
    direction = ["straight", "stop", "left", "right"]
    count = 0
    test_info = []
    track_list = list(tracks_json.keys())
    for key, value in queries_json.items():
        keywords[key] = []
        img_list = []
        track_info = tracks_json[track_list[count]]
        img_list = [track_list[count], len(track_info['frames'])]
        bbox_list_y = []
        bbox_list_x = []
        for bbox in track_info['boxes']:
            bbox_list_y.append(bbox[1]+bbox[3])
            bbox_list_x.append([bbox[0]+bbox[2]])
        count += 1
        reg = 0
        for nl_lines in [value["nl"]]:
            for idx, text in enumerate(nl_lines):
                for direct in direction_dict.keys():
                    if direct in text:
                        keywords[key].append(direct)
                        break
        new_direct = keywords[key].copy()
        if "stop" in new_direct:
            new_direct.remove("stop")
            if len(new_direct) == 0:
                new_direct = keywords[key].copy()
        new_new_direct = []
        for direction_label in new_direct:
            new_new_direct.append(direction[direction_dict[direction_label]])
        direction_label = mode(new_new_direct)
        not_move = [0]
        pre = [bbox_list_x[0][0], bbox_list_y[1]]
        for x, y in zip(bbox_list_x[1:], bbox_list_y[1:]):
            if abs(pre[0] - x[0]) + abs(pre[1] - y) < 3:
                not_move[-1] += 1
            else:
                not_move.append(0)
            pre = [x[0], y]
        x_start, y_start = bbox_list_x[:20], bbox_list_y[:20]
        model = LinearRegression()
        model.fit(x_start, y_start)
        v_start = np.array([model.intercept_, model.coef_[0]])
        x_end, y_end = bbox_list_x[-20:], bbox_list_y[-20:]
        model.fit(x_end, y_end)
        v_end = np.array([model.intercept_, model.coef_[0]])
        angle = np.math.atan2(np.linalg.det([v_start,v_end]),np.dot(v_start,v_end))
        deg = np.degrees(angle)
        pred = ""
        if -10 >= deg:
            pred = "right"
        elif 10 <= deg:
            pred = "left"
        elif max(not_move) > 80 and max(not_move)/len(bbox_list_y) > 0.8:
            pred = "stop"
        elif -10 < deg < 5:
            pred = "straight"
        new_direct_lines += pred + " " + direction[direction_dict[direction_label]] + " " + track_list[count-1] + " " + key + "\n"
    with open(output_path, "w") as w:
        w.write(new_direct_lines)
        
    
def motion_matrix(path):     
    id_motion = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id_motion.append(line.strip().split()[0])
    print(id_motion)
    motion_matrix = np.zeros((184, 184))
    for idx, iv in enumerate(id_motion):
        for jdx, jv in enumerate(id_motion):
            if iv == jv:
                motion_matrix[idx][jdx] = 1
    np.save("vmm.npy", motion_matrix)

    
output_path = "pred_text_video.txt"
motion_test_info(output_path)    
motion_matrix(output_path)