import json
import numpy as np
import math

root_path = "/project/kpsounis_171"
root_path = "/scratch/YT_dataset"
# VERSION = "final"
# TAG = "_filter"
VERSION = "40_June"
# VERSION = "realuser_all"
# VERSION = "reddit_40_new"
TAG = "tags"
TAG = ""

with open(f"{root_path}/dataset/video_ids_{VERSION}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)

with open(f"{root_path}/dataset/video_metadata_{VERSION}.json", "r") as json_file:
    VIDEO_METADATA = json.load(json_file)

with open(f"{root_path}/dataset/video_metadata_40_June.json", "r") as json_file:
    VIDEO_METADATA.update(json.load(json_file))
    
ID2VIDEO = {}
for video_id in VIDEO_IDS.keys():
    ID2VIDEO[str(VIDEO_IDS[video_id])] = video_id

with open(f"{root_path}/dataset/topic/tag2class_{VERSION}2.json", "r") as json_file:
    TAG_CLASS = json.load(json_file)

with open(f"{root_path}/dataset/topic/tag2class_40_June2.json", "r") as json_file:
    TAG_CLASS.update(json.load(json_file))

VIDEO_CLASS = {}
for video_id in VIDEO_METADATA.keys():
    classes = {}
    classes_conf = {}
    try:
        tags = VIDEO_METADATA[video_id]["tags"].split(",")

        for tag in tags:
            if tag == "":
                continue
            try:
                c = TAG_CLASS[tag][0]
                if c not in classes.keys():
                    classes[c] = 0
                classes[c] += 1
            except:
                continue
        
        VIDEO_CLASS[video_id] = ""

        max_c = 0
        for c in classes.keys():
            if classes[c] > max_c:
                VIDEO_CLASS[video_id] = c
                max_c = classes[c]
        
    except:
        continue

VIDEO_BY_CATE = {}

for video_id in VIDEO_METADATA.keys():
    try:
        cate =  VIDEO_CLASS[video_id]
        if cate not in VIDEO_BY_CATE.keys():
            VIDEO_BY_CATE[cate] = []
        VIDEO_BY_CATE[cate].append(VIDEO_IDS[video_id])
    except:
        continue

# for cate in VIDEO_BY_CATE.keys():
#     print(cate, len(VIDEO_BY_CATE[cate]), VIDEO_BY_CATE[cate][0:10])

with open(f"./results/bias_weight_new.json", "r") as json_file:
    BIAS_WEIGHT = json.load(json_file)

with open(f"../dataset/video_adj_list_final_w.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)

def kl_divergence(p, q):
	return sum([p[i] * np.log2((p[i] + 1e-9)/(q[i] + 1e-9)) for i in range(len(p))])
    # return math.sqrt(sum([(p[i] - q[i]) ** 2 for i in range(len(p))]))

def ekl_divergence(p, q):
    for i in range(len(p)):
        if p[i] == 0:
            p[i] += 1e-7

    for i in range(len(p)):
        p[i] /= sum(p)

    return -sum([p[i] * np.log2((p[i])/(1 / 154)) for i in range(len(p))])

SENSITIVITY_W = [1 for _ in range(3)] + [154/27 for _ in range(3)] + [1 for _ in range(101)] + [154/27 for _ in range(24)] + [1 for _ in range(23)]
def wkl_divergence(p, q):
    undesired_dist = 0
    desired_dist = 0
    for i in range(len(p)):
        if SENSITIVITY_W[i] > 1:
            if p[i] < 1e-4:
                p[i] = 1e-4
            undesired_dist += p[i] * np.log2((p[i] * 1)/(q[i] * 0 + 1e-4))
        else:
            desired_dist += p[i] * np.log2((p[i] + 1e-4)/(q[i] + 1e-4))
    return (desired_dist, undesired_dist)