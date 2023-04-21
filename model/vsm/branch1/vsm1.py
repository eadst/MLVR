import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import cv2
import os
import json


def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
#     response = requests.get(url)
    pil_image = Image.open(url).convert("RGB")
    # convert to BGR format
    # newsize = (480, 480)
    # pil_image = pil_image.resize(newsize)
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
    
    
    
def test_track():
    path = "/data/dong/aicity23/data/test-tracks.json"
    data = {}
    images = []
    with open(path, "r") as f:
        data = json.load(f)
    suffix = "/data/dong/aicity23/data"
    print(len(data))
    for idx, vid in enumerate(data):
        info = data[vid]
        roi = os.path.dirname(info['frames'][0])[:-4] + "roi.jpg"
        roi_full_path = os.path.join(suffix, roi)
        mask_roi = cv2.imread(roi_full_path)
        video_clip = []
        img_path = info['frames'][-10]
        img_full_path = os.path.join(suffix, img_path)
        # = cv2.imread(img_full_path)
        #img_mask = cv2.bitwise_and(img, mask_roi) 
        images.append([vid, img_full_path])
    return images
    
    
def test_caption():
    txt_path = "/data/dong/aicity23/data/test-queries.json"
    txt_data = {}
    with open(txt_path, "r") as f:
        txt_data = json.load(f)
    print(len(txt_data))
    texts = []
    aicity_dict = {'sentences':[]}
    for tid in txt_data:
        info = txt_data[tid]
        count = 0
        length_max = 0
        longest = ""
        for keyword in [info['nl'], info['nl_other_views']]:
            for fidx, nl_item in enumerate(keyword):
                if length_max < len(nl_item):
                    length_max = len(nl_item)
                    longest = nl_item
        texts.append([tid, longest])
    return texts
    
    
def test_caption2():
    txt_path = "test_info.txt"
    with open(txt_path, "r") as f:
        txt_data = f.readlines()
    print(len(txt_data))
    texts = []
    for info in txt_data:
        att = info.strip().split()
        text = att[2] + " " + att[3] + ". " + att[5] + " " + att[6] + "."
        texts.append([att[0], text])
    return texts
    
        
# Use this command for evaluate the GLPT-T model
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"


# update the config options with the config file
# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=1080,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)


def main(captions, images):    
    matrix = np.zeros((184, 184))
    total = len(images)
    for idx, img in enumerate(images):
        image = load(img[1])
        for tdx, text in enumerate(captions):
            result, top_predictions, boxes, scores, new_labels = glip_demo.run_on_web_image(image, text[1], 0.7)    
            # print(boxes, scores, new_labels)
            labels = top_predictions.get_field("labels")
            if len(new_labels) > 0:
                matrix[tdx][idx] += len(new_labels)
    np.save("vs1.npy", matrix)
    print(matrix)
        
 
 
images = test_track()
captions = test_caption2()
main(captions, images)

    
    