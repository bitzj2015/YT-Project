import json
import numpy as np
import math

root_path = "/project/kpsounis_171"
root_path = "/scratch/YT_dataset"
# VERSION = "final"
# TAG = "_filter"
VERSION = "40_June"
TAG = ""

with open(f"{root_path}/dataset/video_ids_{VERSION}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)
    
ID2VIDEO = {}
for video_id in VIDEO_IDS.keys():
    ID2VIDEO[str(VIDEO_IDS[video_id])] = video_id
print(len(ID2VIDEO.keys()))

with open(f"./results/bias_weight_new.json", "r") as json_file:
    BIAS_WEIGHT = json.load(json_file)

with open(f"../dataset/video_adj_list_final_w.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)

def kl_divergence(p, q):
	return sum([p[i] * np.log2(p[i]/q[i]) for i in range(len(p))])
    # return math.sqrt(sum([(p[i] - q[i]) ** 2 for i in range(len(p))]))