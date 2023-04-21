import os
import cv2


def get_imgs(path):
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
        suffix = "./aicity23/data"
        print("total tracks: ", len(data))
    for idx, vid in enumerate(data):
        info = data[vid]
        for fidx, img_path in enumerate(info['frames']):
            box = info['boxes'][fidx]
            img_full_path = os.path.join(suffix, img_path)
            img = cv2.imread(img_full_path)
            save_img_path = "./data/vct_data/crop_{}_{}.jpg".format(vid, fidx)
            cv2.imwrite(save_img_path, img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])            

            
path = "./aicity23/data/train-tracks.json"
get_imgs()
path = "./aicity23/data/test-tracks.json"
get_imgs()