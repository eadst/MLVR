import sys
import json
import en_core_web_sm
from statistics import mode
import numpy as np


# get vihecle color and type labels
def get_vehicle_train_info():
    train_path = "./aicity23/data/train-tracks.json"
    with open(train_path) as f:
        train_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    new_lines = ""
    new_lines2 = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10, "Na": 11}
    color_color = ["blue", "brown", "gray", "orange", "black", "purple", "silver", "green", "white", "yellow", "red", "Na"]
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "wagon":0, "coupe":1, "sedan":1, "car": 1, "van":2, "minivan":2, "bus":2, "mpv":2, "pickup":3, "truck":3, "Na": 4}
    type_type = ["suv", "car", "van", "pickup", "Na"]
    for key, value in train_json.items():
        keywords[key] = [[], []]
        img_list = []
        for fidx, img_path in enumerate(value['frames']):
            save_img_path = "./data/vct_data/crop_{}_{}.jpg".format(key, fidx)
            img_list.append(save_img_path)
        for nl_lines in [value["nl"], value["nl_other_views"]]:
            for idx, text in enumerate(nl_lines):
                extractor = nlp(text)
                for chunk in extractor.noun_chunks:
                    for word in chunk:
                        lword = str(word).lower()
                        if word.pos_ == 'ADJ':
                            if lword in color_dict:
                                keywords[key][0].append(lword)
                        if lword in type_dict:
                            keywords[key][1].append(lword)
                            break
                    break
        try:
            # get the mode class of the list
            color_label, type_label = mode(keywords[key][0]), mode(keywords[key][1])
            
            for img in img_list:
                new_lines += img + " " + color_color[color_dict[color_label]] + "\n"
                new_lines2 += img + " " + type_type[type_dict[type_label]] + "\n"
        except:
            print(key, keywords[key])
            
    # save the path and label        
    with open("vehicle_color_train.txt", "w") as w:
        w.write(new_lines)
    with open("vehicle_type_train.txt", "w") as w:
        w.write(new_lines2)
        
        
get_vehicle_train_info()