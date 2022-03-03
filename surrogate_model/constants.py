import json
import numpy as np

VERSION = "final"
TAG = "_filter"
ROOT_PATH = "/project/kpsounis_171"
with open(f"{ROOT_PATH}/dataset/home_video_id_sorted_{VERSION}{TAG}.json", "r") as json_file:
    data = json.load(json_file)
home_video_id_sorted = [int(key) for key in data.keys()]
all_values = list(data.values())
home_video_value_sorted = [int(key) / sum(data.values()) for key in all_values]

with open(f"{ROOT_PATH}/dataset/video2channel_ids_{VERSION}.json", "r") as json_file:
    video2channel_ids = json.load(json_file)
    
with open(f"{ROOT_PATH}/dataset/video_ids_{VERSION}.json", "r") as json_file:
    video_ids = json.load(json_file)
    
video2channel = {}
for video_id in video2channel_ids.keys():
    video2channel[video_ids[video_id]] = video2channel_ids[video_id]
    
with open(f"{ROOT_PATH}/dataset/video_adj_list_{VERSION}_w.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)

def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))