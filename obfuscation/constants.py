import json
import numpy as np


VERSION = "reddit"
TAG = "_filter"
with open(f"../dataset/video_ids_{VERSION}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)
    
ID2VIDEO = {}
for video_id in VIDEO_IDS.keys():
    ID2VIDEO[str(VIDEO_IDS[video_id])] = video_id
print(len(ID2VIDEO.keys()))

with open(f"./results/bias_weight_new.json", "r") as json_file:
    BIAS_WEIGHT = json.load(json_file)

def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))