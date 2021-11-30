import json
from tqdm import tqdm
import h5py
import numpy as np

VERSION = "rand_new"
with open(f"../dataset/sock_puppets_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]

# with open(f"../dataset/sock_puppets_reddit.json", "r") as json_file:
#     data_truth = json.load(json_file)[2]["data"]

with open(f"../dataset/video_ids_{VERSION}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)

with h5py.File(f"../dataset/video_embeddings_{VERSION}.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")


print(len(data))

initial_home_video_ids = {}
unique_home_video_id = {}

data_truth = data
for i in tqdm(range(len(data_truth))):
    initial_home_video_ids.update(dict(zip(data_truth[i]["initial_homepage"], [1 for _ in range(len(data_truth[i]["initial_homepage"]))])))
    video_views = data_truth[i]["homepage"]
    tmp = {}
    for video_view in video_views:
        for video_id in video_view:
            if video_id in initial_home_video_ids.keys():
                continue
            if video_id not in unique_home_video_id.keys():
                unique_home_video_id[video_id] = 0
            if video_id not in tmp.keys():
                tmp[video_id] = 1
                unique_home_video_id[video_id] += 1
            else:
                tmp[video_id] = 1
print(len(initial_home_video_ids))
removed_videos = []
for video in unique_home_video_id.keys():
    if unique_home_video_id[video] > 10 and unique_home_video_id[video] < 1000:
        removed_videos.append(video)
for video in removed_videos:
    del unique_home_video_id[video]

print(len(unique_home_video_id))
unique_home_video_id = {}
# initial_home_video_ids = {}

avergea_r = 0
N = 150
k = 1
similarity = 0
for i in range(N):
    base_data = data[i]["homepage"]
    base_home_video_ids = {}
    for home_rec in base_data:
        for video_id in home_rec:
            if video_id in initial_home_video_ids.keys():
                continue
            if video_id in unique_home_video_id.keys():
                continue
            if video_id not in base_home_video_ids.keys():
                base_home_video_ids[video_id] = 0
            base_home_video_ids[video_id] += 1
    base_home_video_ids = {VIDEO_IDS[k] : v for k, v in sorted(base_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
    base_home_video_ids_list = list(base_home_video_ids.keys())
    base_emb = np.mean(video_embeddings[base_home_video_ids_list], axis=0)
    
    obfu_data = data[i+N]["homepage"]
    obfu_home_video_ids = {}
    for home_rec in obfu_data:
        for video_id in home_rec:
            if video_id in initial_home_video_ids.keys():
                continue
            if video_id in unique_home_video_id.keys():
                continue
            if video_id not in obfu_home_video_ids.keys():
                obfu_home_video_ids[video_id] = 0
            obfu_home_video_ids[video_id] += 1
    obfu_home_video_ids = {VIDEO_IDS[k] : v for k, v in sorted(obfu_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
    obfu_home_video_ids_list = list(obfu_home_video_ids.keys())
    obfu_emb = video_embeddings[obfu_home_video_ids_list]
    obfu_emb = np.mean(video_embeddings[base_home_video_ids_list], axis=0)

    similarity += np.sum(base_emb * obfu_emb)

    # print(data[i]["viewed"])
    # print(data[i+N]["viewed"])
    # print(sorted(base_home_video_ids))
    # print(sorted(obfu_home_video_ids))
    # seed_video = data[i]["viewed"][0]
    # for item in data_truth:
    #     # if item["seedId"] == seed_video and item["viewed"][1] == data[i]["viewed"][1]:
    #     if item["viewed"][0] == data[i]["viewed"][0] and item["viewed"][1] == data[i]["viewed"][1]:
    #         item_home_video_ids = {}
    #         for home_rec in item["homepage"]:
    #             for video_id in home_rec:
    #                 if video_id in initial_home_video_ids.keys():
    #                     continue
    #                 if video_id in unique_home_video_id.keys():
    #                     continue
    #                 if video_id not in item_home_video_ids.keys():
    #                     item_home_video_ids[video_id] = 0
    #                 item_home_video_ids[video_id] += 1
    #         item_home_video_ids = {k : v for k, v in sorted(item_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
            # print(sorted(item_home_video_ids))
    # break
  
    removed = 0
    added = 0
    for video in base_home_video_ids.keys():
        if video not in obfu_home_video_ids.keys():
            removed += 1
    for video in obfu_home_video_ids.keys():
        if video not in base_home_video_ids.keys():
            added += 1
    avergea_r += (removed + added)

print(avergea_r / N, similarity / N)

