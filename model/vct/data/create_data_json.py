import json


def get_json(all_cls, path_dict, output):
    json_dict = {"train": [], "val": [], "test": []}
    for key, path in path_dict.items():
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split()
                idx = all_cls.index(all_cls[int(info[1])])
                json_dict[key].append([info[0], idx, all_cls[idx]])
    with open(output, 'w') as f:
        json.dump(json_dict, f)

        
# create vehicle color json file
all_cls = ["blue car", "brown car", "gray car", "orange car", "black car", "purple car", "silver car", "green car", "white car", "yellow car", "red car"]
path_dict = {"train": "./vehicle_color_train.txt", 
             "val": "./vehicle_color_val.txt", 
             "test": "./vehicle_color_test.txt"}
output = 'color_data.json'
get_json(all_cls, path_dict, output)

# create vehicle type json file
all_cls = ["suv", "car", "van", "pickup", "cargo", "bus"]
path_dict = {"train": "./vehicle_type_train.txt", 
             "val": "./vehicle_type_val.txt", 
             "test": "./vehicle_type_test.txt"}
output = 'type_data.json'
get_json(all_cls, path_dict, output)
