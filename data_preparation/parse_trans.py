import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from constants import *

def filter_trans(video_id: str, file_path: str, output_path: str):
    lines = []
    res = []
    with open(file_path, "r",encoding="ISO-8859-1") as text_file:
        for line in text_file.readlines():
            if line.endswith("</c>\n"):
                lines.append(line)

    for line in lines:
        res += [i.split("<")[0] for i in line.split("<c> ")]

    if len(res) == 0:
        lines = []
        res = []
        with open(file_path, "r") as text_file:
            for line in text_file.readlines():
                if line == "\n":
                    continue
                elif len(line.split('-->')) == 2:
                    continue
                else:
                    lines.append(line[:-1].replace(',', '').replace('.', '').lower())
        for line in lines:
            res += line.split(" ")
    with open(output_path, "w") as json_file:
        json.dump({video_id: " ".join(res)}, json_file)

    return len(res)

len_dist = {}
parsed_video_ids = {}
for filename in os.listdir(f"{root_path}/dataset/trans_parsed"):
    video_id = filename.split(".")[0]
    parsed_video_ids[video_id] = 1

for filename in tqdm(os.listdir(f"{root_path}/dataset/trans_en")):
    video_id = filename.split(".")[0]
    file_path = f"{root_path}/dataset/trans_en/{filename}"
    output_path = f"{root_path}/dataset/trans_parsed/{video_id}.json"
    if video_id in parsed_video_ids.keys():
        continue
    cur_len = filter_trans(video_id, file_path, output_path)
    if cur_len not in len_dist.keys():
        len_dist[cur_len] = 0
    len_dist[cur_len] += 1

len_dist = dict(sorted(len_dist.items()))
with open("./logs/len_dist.json", "w") as json_file:
    json.dump(len_dist, json_file)

import numpy as np

with open("./logs/len_dist.json", "r") as json_file:
    len_dist = json.load(json_file)

x = np.log10(np.array([int(key) for key in len_dist.keys()]))
y = np.array(list(len_dist.values()))
y = y/np.sum(y)
y = np.matmul(y.reshape(1,-1), np.triu(np.ones(y.shape))).reshape(-1)
plt.figure()
plt.plot(x,y,"*-")
plt.xlabel("Transcript length (log_10)")
plt.ylabel("CDF")
plt.savefig("./fig/len_dist.png")

    