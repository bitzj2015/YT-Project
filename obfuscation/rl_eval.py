import json
from tqdm import tqdm

VERSION = "rl"
with open(f"../dataset/sock_puppets_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]

with open(f"../dataset/sock_puppets_final.json", "r") as json_file:
    data_truth = json.load(json_file)[2]["data"]

print(len(data))

initial_home_video_ids = {}
for i in tqdm(range(len(data_truth))):
    try:
        initial_home_video_ids.update(dict(zip(data_truth[i]["initial_homepage"], [1 for _ in range(len(data_truth[i]["initial_homepage"]))])))
    except: 
        continue
print(len(initial_home_video_ids))

avergea_r = 0
for i in range(150):
    base_data = data[i]["homepage"]
    base_home_video_ids = {}
    for home_rec in base_data:
        for video_id in home_rec:
            if video_id in initial_home_video_ids.keys():
                continue
            if video_id not in base_home_video_ids.keys():
                base_home_video_ids[video_id] = 0
            base_home_video_ids[video_id] += 1
    base_home_video_ids = {k : v for k, v in sorted(base_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}

    obfu_data = data[i + 150]["homepage"]
    obfu_home_video_ids = {}
    for home_rec in obfu_data:
        for video_id in home_rec:
            if video_id in initial_home_video_ids.keys():
                continue
            if video_id not in obfu_home_video_ids.keys():
                obfu_home_video_ids[video_id] = 0
            obfu_home_video_ids[video_id] += 1
    obfu_home_video_ids = {k : v for k, v in sorted(obfu_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
    print(data[i]["viewed"], len(data[i]["viewed"]))
    print(data[i + 150]["viewed"], len(data[i + 150]["viewed"]))
    print(sorted(base_home_video_ids))
    print(sorted(obfu_home_video_ids))
    seed_video = data[i]["viewed"][0]
    for item in data_truth:
        if item["seedId"] == seed_video and item["viewed"][1] == data[i]["viewed"][1]:
            item_home_video_ids = {}
            for home_rec in item["homepage"]:
                for video_id in home_rec:
                    if video_id in initial_home_video_ids.keys():
                        continue
                    if video_id not in item_home_video_ids.keys():
                        item_home_video_ids[video_id] = 0
                    item_home_video_ids[video_id] += 1
            item_home_video_ids = {k : v for k, v in sorted(item_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
            print(sorted(item_home_video_ids))
    # break
  
    removed = 0
    added = 0
    for video in base_home_video_ids.keys():
        if video not in item_home_video_ids.keys():
            removed += 1
    for video in item_home_video_ids.keys():
        if video not in base_home_video_ids.keys():
            added += 1
    avergea_r += (removed + added)
    break
print(avergea_r / 150)

