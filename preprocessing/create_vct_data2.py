import json
import sys
import en_core_web_sm
from statistics import mode
from sklearn.linear_model import LinearRegression
import numpy as np

def get_vehicle_info():
    train_path = "./aicity23/data/train-tracks.json"
    with open(train_path) as f:
        train_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    new_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10}
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "coupe":1, "van":2, "bus":2, "mpv":2, "pickup":2, "truck":3, "sedan":4, "wagon":5}
    for key, value in train_json.items():
        keywords[key] = [[], []]
        img_list = []
        for fidx, img_path in enumerate(value['frames']):
            save_img_path = "./aicity/crop_{}_{}.jpg".format(key, fidx)
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
            color_label, type_label = mode(keywords[key][0]), mode(keywords[key][1])
            
            for img in img_list:
                new_lines += img + " " + color_label + " " + type_label + "\n"
        except:
            print(key, keywords[key])
#         break
            
#     with open("id_color_type.json", "w") as f:
#         json.dump(mode_keywords, f)
    with open("id_color_type.txt", "w") as w:
        w.write(new_lines)


def get_test_vehicle_info():
    tracks_path = "/dfs/data/others/aicity23/data/test-tracks.json"
    with open(tracks_path) as f:
        tracks_json = json.load(f)  
    queries_path = "/dfs/data/others/aicity23/data/test-queries.json"
    with open(queries_path) as f:
        queries_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    id_color = {}
    new_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10}
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "wagon":0, "coupe":1, "sedan":1, "car": 1, "van":2, "minivan":2, "bus":2, "mpv":2, "pickup":3, "truck":3}
    for key, value in queries_json.items():
#         key = "aa7eda10-2233-44ab-8542-b02723107f46"
#         value = queries_json[key]
#         print(value)
#         break
        keywords[key] = [[], []]
        img_list = []
#         track_info = tracks_json[key]
#         for fidx, img_path in enumerate(track_info['frames']):
#             save_img_path = "./aicity_test/crop_{}_{}.jpg".format(key, fidx)
#             img_list.append(save_img_path)
        for nl_lines in [value["nl"], value["nl_other_views"]]:
            for idx, text in enumerate(nl_lines):
                extractor = nlp(text)
                for chunk in extractor.noun_chunks:
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][0].append(lword)
                        if lword in type_dict:
                            keywords[key][1].append(lword)
                            break
                    break
        try:
            color_label, type_label = mode(keywords[key][0]), mode(keywords[key][1])
            #print(color_label, type_label)
            id_color[key] = color_dict[color_label]
#             id_color[key] = type_dict[type_label]
#             for img in img_list:
#                 new_lines += img + " " + color_label + " " + type_label + "\n"
        except:
            print(key, keywords[key])
#         break
#     print(id_color)
    with open("id_color_test.json", "w") as f:
        json.dump(id_color, f)
    import numpy as np
    color_matrix = np.zeros((184, 184))
    for idx, iv in enumerate(id_color):
        for jdx, jv in enumerate(id_color):
            if id_color[iv] == id_color[jv]:
                color_matrix[idx][jdx] = 1
    np.save("id_color_matrix.npy", color_matrix)
#     with open("id_color_type_test.txt", "w") as w:
#         w.write(new_lines)

# get_vehicle_info()
# get_test_vehicle_info()

def get_vehicle_train_info():
    train_path = "/dfs/data/others/aicity23/data/train-tracks.json"
    with open(train_path) as f:
        train_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    new_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10}
    color_color = ["blue", "brown", "gray", "orange", "black", "purple", "silver", "green", "white", "yellow", "red"]
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "coupe":1, "van":2, "bus":2, "mpv":2, "pickup":2, "truck":3, "sedan":4, "wagon":5}
    for key, value in train_json.items():
#         key = "e4adcbc2-559a-41c9-88c4-ec563c7816f4"
#         value = train_json[key]
        keywords[key] = [[], []]
        img_list = []
        for fidx, img_path in enumerate(value['frames']):
            save_img_path = "./aicity/crop_{}_{}.jpg".format(key, fidx)
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
            color_label, type_label = mode(keywords[key][0]), mode(keywords[key][1])
            
            for img in img_list:
                new_lines += img + " " + color_color[color_dict[color_label]] + "\n"
#                 new_lines += img + " " + color_label + " " + type_label + "\n"
        except:
            print(key, keywords[key])
#         break
            
#     with open("id_color_type.json", "w") as f:
#         json.dump(mode_keywords, f)
    with open("id_color_train.txt", "w") as w:
        w.write(new_lines)

        
        
def get_vehicle_test_info(output_path):
    tracks_path = "/dfs/data/others/aicity23/data/test-tracks.json"
    with open(tracks_path) as f:
        tracks_json = json.load(f)  
    queries_path = "/dfs/data/others/aicity23/data/test-queries.json"
    with open(queries_path) as f:
        queries_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    id_color = {}
    new_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10, "Na": 11}
    color_color = ["blue", "brown", "gray", "orange", "black", "purple", "silver", "green", "white", "yellow", "red", "Na"]
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "wagon":0, "coupe":1, "sedan":1, "car": 1, "van":2, "minivan":2, "bus":2, "mpv":2, "pickup":3, "truck":3, "Na": 4}
    type_type = ["suv", "car", "van", "pickup", "Na"]
    direction_dict = {"straight": 0, "down": 0, "through": 0, "stop": 1, "left": 2, "right": 3}
    direction = ["straight", "stop", "left", "right"]
#     direction_dict = {"straight": 0, "down": 0, "through": 0, "left": 1, "right": 2}
#     direction = ["straight", "left", "right"]
    neighbors = ["by"]
    count = 0
    test_info = []
    track_list = list(tracks_json.keys())
    for key, value in queries_json.items():
        keywords[key] = [[], [], [], ["Na"], ["Na"]]
        img_list = []
        track_info = tracks_json[track_list[count]]
        img_list = [track_list[count], len(track_info['frames'])]
        count += 1
        reg = 0
        for nl_lines in [value["nl"], value["nl_other_views"]]:
            for idx, text in enumerate(nl_lines):
                for direct in direction_dict.keys():
                    if direct in text:
                        keywords[key][2].append(direct)
                extractor = nlp(text)
                nouns = list(extractor.noun_chunks)
                if len(nouns) > 0:
                    chunk = list(extractor.noun_chunks)[0]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][0].append(lword)
                        if lword in type_dict:
                            keywords[key][1].append(lword)
                            break
                if len(nouns) > 1:
                    chunk = list(extractor.noun_chunks)[-1]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][3].insert(0, lword)
                        if lword in type_dict:
                            keywords[key][4].insert(0, lword)
                            break
        try:
            new_type = keywords[key][1].copy()
            if "car" in new_type:
                new_type.remove("car")
                if len(new_type) == 0:
                    new_type = keywords[key][1].copy()
            new_direct = keywords[key][2].copy()
            if "stop" in new_direct:
                new_direct.remove("stop")
                if len(new_direct) == 0:
                    new_direct = keywords[key][2].copy()
            color_label, type_label, direction_label = mode(keywords[key][0]), mode(new_type), mode(new_direct)
            nb_color_label, nb_type_label = mode(keywords[key][3]), mode(keywords[key][4])
            new_lines += img_list[0] + " " + str(img_list[1]) + " " \
                        + color_color[color_dict[color_label]] + " " \
                        + type_type[type_dict[type_label]] + " " \
                        + direction[direction_dict[direction_label]] + " " \
                        + color_color[color_dict[nb_color_label]] + " " \
                        + type_type[type_dict[nb_type_label]] + " " + track_list[count-1] + "\n"
            id_color[key] = direction[direction_dict[direction_label]]
#             id_color[key] = color_color[color_dict[nb_color_label]]
#             id_color[key] = type_type[type_dict[nb_type_label]]
#             if track_list[count-1] == "ed81eaa0-6b0d-411f-91a9-667facde7800":
#                 print("++++++++++", key)
        except:
            print(key, keywords[key])
            print(value["nl"], value["nl_other_views"])
            print(track_list[count-1], count-1)

    with open(output_path, "w") as w:
        w.write(new_lines)

#     import numpy as np
#     color_matrix = np.zeros((184, 184))
#     for idx, iv in enumerate(id_color):
#         for jdx, jv in enumerate(id_color):
#             if id_color[iv] == id_color[jv]:
#                 color_matrix[idx][jdx] = 1
#     np.save("id_direct_matrix.npy", color_matrix)
#     print(color_matrix)

        
def get_vehicle_train_info(output_path):
    train_path = "/dfs/data/others/aicity23/data/train-tracks.json"
    with open(train_path) as f:
        train_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    id_color = {}
    new_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10, "Na": 11}
    color_color = ["blue", "brown", "gray", "orange", "black", "purple", "silver", "green", "white", "yellow", "red", "Na"]
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "wagon":0, "coupe":1, "sedan":1, "car": 1,  "vehicle": 1, "van":2, "minivan":2, "bus":2, "mpv":2, "pickup":3, "truck":3, "Na": 4}
    type_type = ["suv", "car", "van", "pickup", "Na"]
    direction_dict = {"straight": 0, "down": 0, "through": 0, "left": 1, "right": 2}
    direction = ["straight", "left", "right"]
#     direction = ["straight", "stop", "left", "right"]
    neighbors = ["by"]
    count = 0
    test_info = []
    for key, value in train_json.items():
        keywords[key] = [[], [], [], ["Na"], ["Na"]]
        img_list = [key, len(value['frames'])]
        for nl_lines in [value["nl"], value["nl_other_views"]]:
            for idx, text in enumerate(nl_lines):
                for direct in direction_dict.keys():
                    if direct in text:
                        keywords[key][2].append(direct)
                extractor = nlp(text)
                nouns = list(extractor.noun_chunks)
                if len(nouns) > 0:
                    chunk = list(extractor.noun_chunks)[0]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][0].append(lword)
                        if lword in type_dict:
                            keywords[key][1].append(lword)
                            break
                if len(nouns) > 1:
                    chunk = list(extractor.noun_chunks)[-1]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][3].insert(0, lword)
                        if lword in type_dict:
                            keywords[key][4].insert(0, lword)
                            break
        try:
            color_label, type_label, direction_label = mode(keywords[key][0]), mode(keywords[key][1]), mode(keywords[key][2])
            nb_color_label, nb_type_label = mode(keywords[key][3]), mode(keywords[key][4])
            if nb_type_label == "Na":
                nb_color_label = "Na"
            new_lines += img_list[0] + " " + str(img_list[1]) + " " \
                        + color_color[color_dict[color_label]] + " " \
                        + type_type[type_dict[type_label]] + " " \
                        + direction[direction_dict[direction_label]] + " " \
                        + color_color[color_dict[nb_color_label]] + " " \
                        + type_type[type_dict[nb_type_label]] + "\n"
        except:
            print(key, keywords[key])

    with open(output_path, "w") as w:
        w.write(new_lines)
        

# output_path = "./text/test_info2.txt"
# get_vehicle_test_info(output_path)
# output_path = "./text/train_info.txt"
# get_vehicle_train_info(output_path)


def direct_test_info(output_path):
    tracks_path = "/dfs/data/others/aicity23/data/test-tracks.json"
    with open(tracks_path) as f:
        tracks_json = json.load(f)  
    queries_path = "/dfs/data/others/aicity23/data/test-queries.json"
    with open(queries_path) as f:
        queries_json = json.load(f)  
    nlp = en_core_web_sm.load()
    keywords = {}
    id_color = {}
    new_lines = ""
    new_direct_lines = ""
    color_dict = {"blue":0, "brown":1, "gray":2, "grey":2, "orange":3, "black":4, "purple":5, "silver":6, "green":7, "white":8, "yellow":9, "red":10, "maroon":10, "reddish":10, "Na": 11}
    color_color = ["blue", "brown", "gray", "orange", "black", "purple", "silver", "green", "white", "yellow", "red", "Na"]
    type_dict = {"suv":0, "hatchback": 0, "jeep":0, "cross-over":0, "wagon":0, "coupe":1, "sedan":1, "car": 1, "van":2, "minivan":2, "bus":2, "mpv":2, "pickup":3, "truck":3, "Na": 4}
    type_type = ["suv", "car", "van", "pickup", "Na"]
    direction_dict = {"straight": 0, "forward":0, "down": 0, "through": 0, "stop": 1, "left": 2, "right": 3}
#     direction_list = ["straight", "down", "through", "stop", "left", "right"]
    direction = ["straight", "stop", "left", "right"]
    del_direction = ["right lane", "left lane", "right line", "left line"]
    neighbors = ["by"]
    count = 0
    test_info = []
    track_list = list(tracks_json.keys())
    for key, value in queries_json.items():
        keywords[key] = [[], [], [], ["Na"], ["Na"]]
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
#         for nl_lines in [value["nl"], value["nl_other_views"]]:
        for nl_lines in [value["nl"]]:
            for idx, text in enumerate(nl_lines):
                for direct in direction_dict.keys():
                    if direct in text:
                        keywords[key][2].append(direct)
                        break
                extractor = nlp(text)
                nouns = list(extractor.noun_chunks)
                if len(nouns) > 0:
                    chunk = list(extractor.noun_chunks)[0]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][0].append(lword)
                        if lword in type_dict:
                            keywords[key][1].append(lword)
                            break
                if len(nouns) > 1:
                    chunk = list(extractor.noun_chunks)[-1]
                    for word in chunk:
                        lword = str(word).lower()
                        if lword in color_dict:
                            keywords[key][3].insert(0, lword)
                        if lword in type_dict:
                            keywords[key][4].insert(0, lword)
                            break
#         try:
#             new_type = keywords[key][1].copy()
#             if "car" in new_type:
#                 new_type.remove("car")
#                 if len(new_type) == 0:
#                     new_type = keywords[key][1].copy()
#             new_direct = keywords[key][2].copy()
#             if "stop" in new_direct:
#                 new_direct.remove("stop")
#                 if len(new_direct) == 0:
#                     new_direct = keywords[key][2].copy()
#             color_label, type_label, direction_label = mode(keywords[key][0]), mode(new_type), mode(new_direct)
#             nb_color_label, nb_type_label = mode(keywords[key][3]), mode(keywords[key][4])
#             new_lines += img_list[0] + " " + str(img_list[1]) + " " \
#                         + color_color[color_dict[color_label]] + " " \
#                         + type_type[type_dict[type_label]] + " " \
#                         + direction[direction_dict[direction_label]] + " " \
#                         + color_color[color_dict[nb_color_label]] + " " \
#                         + type_type[type_dict[nb_type_label]] + " " + track_list[count-1] + "\n"
#             id_color[key] = direction[direction_dict[direction_label]]
#         except:
#             print(key, keywords[key])
#             print(value["nl"], value["nl_other_views"])
#             print(track_list[count-1], count-1)
        new_direct = keywords[key][2].copy()
        if "stop" in new_direct:
            new_direct.remove("stop")
            if len(new_direct) == 0:
                new_direct = keywords[key][2].copy()
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
        print("stop: ", max(not_move))
        x_start, y_start = bbox_list_x[:20], bbox_list_y[:20]
        model = LinearRegression()
        model.fit(x_start, y_start)
        v_start = np.array([model.intercept_, model.coef_[0]])
        x_end, y_end = bbox_list_x[-20:], bbox_list_y[-20:]
        model.fit(x_end, y_end)
        v_end = np.array([model.intercept_, model.coef_[0]])
        print(v_start, v_end)
        angle = np.math.atan2(np.linalg.det([v_start,v_end]),np.dot(v_start,v_end))
        deg = np.degrees(angle)
        print("degree: ", deg, max(not_move))
        print(new_direct)
        pred = ""
        if -10 >= deg:
            pred = "right"
        elif 10 <= deg:
            pred = "left"
        elif max(not_move) > 80 and max(not_move)/len(bbox_list_y) > 0.8:
            pred = "stop"
        elif -10 < deg < 5:
            pred = "straight"
#         if pred != direction[direction_dict[direction_label]]:
        if 1:
#             print(pre, direction[direction_dict[direction_label]])
            new_direct_lines += pred + " " + direction[direction_dict[direction_label]] + " " + track_list[count-1] + " " + key + "\n"
    with open(output_path, "w") as w:
        w.write(new_lines)
    with open("./text/test_info4.txt", "w") as w:
        w.write(new_direct_lines)
        
# output_path = "./text/test_info3.txt"
# direct_test_info(output_path)        
def direct_matrix():     
    path = "./text/test_info4.txt"
    id_color = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            id_color.append(line.strip().split()[0])
    print(id_color)
    color_matrix = np.zeros((184, 184))
    for idx, iv in enumerate(id_color):
        for jdx, jv in enumerate(id_color):
            if iv == jv:
                color_matrix[idx][jdx] = 1
            else:
                print(iv, jv)
    np.save("id_direct_matrix_v2.npy", color_matrix)
    print(color_matrix)
    
# direct_matrix()