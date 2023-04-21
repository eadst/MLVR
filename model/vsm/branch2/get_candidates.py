import os
import json

def get_imgs(path):
    data = {}
    path_dict = {} # save the candidates in the same image
    image_dict = {} # save vid all image paths
    with open(path, "r") as f:
        data = json.load(f)
        suffix = "/dfs/data/others/aicity23/data"
        print("total tracks: ", len(data))
    for idx, vid in enumerate(data):
        info = data[vid]
        if vid not in image_dict:
            image_dict[vid] = [] 
        for fidx, img_path in enumerate(info['frames']):
            box = info['boxes'][fidx]
            img_full_path = os.path.join(suffix, img_path)
            if img_full_path not in path_dict:
                path_dict[img_full_path] = [] 
            save_img_path = "./data/vct_data/crop_{}_{}.jpg".format(vid, fidx)
            path_dict[img_full_path].append(save_img_path) 
            image_dict[vid].append(img_full_path) 
    with open("path_dict.json", "w") as f:
        json.dump(path_dict, f)
    with open("image_dict.json", "w") as f:
        json.dump(image_dict, f)
            

            
path = "/dfs/data/others/aicity23/data/test-tracks.json"
get_imgs(path)