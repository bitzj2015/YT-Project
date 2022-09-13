import json
import numpy as np

VERSION = "40_June"
TAG = "tags"
ROOT_PATH = "/project/kpsounis_171"
ROOT_PATH = "/scratch/YT_dataset"

with open(f"{ROOT_PATH}/dataset/home_video_id_sorted_{VERSION}{TAG}.json", "r") as json_file:
    data = json.load(json_file)

# home_video_id_sorted = [int(key) for key in data.keys()]
# all_values = list(data.values())
# all_values_sum = sum(all_values)
# home_video_value_sorted = [int(key) / all_values_sum for key in all_values]

with open(f"{ROOT_PATH}/dataset/video2channel_ids_{VERSION}.json", "r") as json_file:
    video2channel_ids = json.load(json_file)
    
with open(f"{ROOT_PATH}/dataset/video_ids_{VERSION}.json", "r") as json_file:
    video_ids = json.load(json_file)
    
video2channel = {}
for video_id in video2channel_ids.keys():
    video2channel[video_ids[video_id]] = video2channel_ids[video_id]
    
with open(f"../dataset/video_adj_list_final_w.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)

def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

with open(f"{ROOT_PATH}/dataset/topic/class_weight.json", "r") as json_file:
    CLASS_WEIGHT = json.load(json_file)["w"]