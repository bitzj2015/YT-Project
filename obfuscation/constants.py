import json

with open("../dataset/video_ids_reddit.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)
    
ID2VIDEO = {}
for video_id in VIDEO_IDS.keys():
    ID2VIDEO[str(VIDEO_IDS[video_id])] = video_id
print(len(ID2VIDEO.keys()))

with open(f"./results/bias_weight_new.json", "r") as json_file:
    BIAS_WEIGHT = json.load(json_file)
