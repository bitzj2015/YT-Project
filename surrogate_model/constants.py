import json
VERSION = "reddit"
TAG = "_filter_p"
with open(f"../dataset/home_video_id_sorted_{VERSION}{TAG}.json", "r") as json_file:
    data = json.load(json_file)
home_video_id_sorted = [int(key) for key in data.keys()]
all_values = list(data.values())
home_video_value_sorted = [int(key) / sum(data.values()) for key in all_values]

with open(f"../dataset/video2channel_ids_{VERSION}.json", "r") as json_file:
    video2channel_ids = json.load(json_file)
    
with open(f"../dataset/video_ids_{VERSION}.json", "r") as json_file:
    video_ids = json.load(json_file)
    
video2channel = {}
for video_id in video2channel_ids.keys():
    video2channel[video_ids[video_id]] = video2channel_ids[video_id]
    
with open(f"../dataset/video_adj_list_{VERSION}_w.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)
