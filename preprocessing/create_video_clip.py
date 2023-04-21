import os
import cv2
import json

        
def save_video(save_clip_path, video_clip):
    image_info=video_clip[0].shape
    height=image_info[0]
    width=image_info[1]
    size=(height,width)
    fps=16
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(save_clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height)) 
    for img in video_clip:
        video.write(img)
        
        
def background_video(path):
    data = {}
    with open(path, "r") as f:
        data = json.load(f)
    suffix = "./aicity23/data"
    save_path = "./aicity23/testvideobox" 
    for idx, vid in enumerate(data):
        info = data[vid]
        # mask roi
        roi = os.path.dirname(info['frames'][0])[:-4] + "roi.jpg"
        roi_full_path = os.path.join(suffix, roi)
        mask_roi = cv2.imread(roi_full_path)
        # background image
        median_100 = os.path.dirname(info['frames'][0])[:-4] + "median_100.jpg"
        median_100_full_path = os.path.join(suffix, median_100)
        img_median = cv2.imread(median_100_full_path)
        # video clip
        video_clip = []
        for fidx, img_path in enumerate(info['frames']):
            back_img = img_median.copy()
            box = info['boxes'][fidx]
            img_full_path = os.path.join(suffix, img_path)
            img = cv2.imread(img_full_path)
            back_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].copy()
            cv2.rectangle(back_img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (55,255,155), 5)
            img_mask = cv2.bitwise_and(back_img, mask_roi) 
            video_clip.append(img_mask)
        save_clip_path = save_path + "/{}.mp4".format(vid)
        save_video(save_clip_path, video_clip) 
        
            
path = "./aicity23/data/train-tracks.json"   
path = "./aicity23/data/test-tracks.json"   
background_video(path)
