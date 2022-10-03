import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import distance

random.seed(0)
root_path = "/scratch/YT_dataset"

VERSION = "final_joint_cate_100_2_test"
# VERSION = "final_joint_cate_100_2_0.1"
VERSION = "final_joint_cate_103_2_test"
VERSION = "reddit_cate_100_2_test"
VERSION = "latest_joint_cate_010"
# VERSION = "latest_joint_cate_010_reddit3_0.2"
# VERSION = "latest_joint_cate_010_0.3"
VERSION = "v1_binary_0.2_test"
# VERSION = "0.3_v2_kldiv_0.3_test_0.2_test"
# VERSION = "realuser_0.2_test"
VERSION = "0.2_v2_kldiv_0.2_test_0.2_test"
VERSION = "0.3_v2_kldiv_0.3_test"
N = 150
PHASE = 0

video_metadata = {}
tag2class = {}
class2id = {}
data = []
cnt = 0
# for VERSION in ["0.3_v2_kldiv_0.3_test", "0.5_v2_kldiv_0.5_test", "0.2_v2_kldiv_0.2_test", "0.2_v2_kldiv_0.2_test_0.2_test", "0.3_v2_kldiv_0.3_test_0.2_test", "0.5_v2_kldiv_0.5_test_0.2_test"]:
# for VERSION in ["realuser", "realuser_0.2_test", "realuser_0.2_test_v2", "realuser_all"]:
for VERSION in ["0.2_v2_kldiv_reddit2_test"]:

    with open(f"{root_path}/dataset/sock_puppets_{VERSION}.json", "r") as json_file:
        data_tmp = json.load(json_file)[2]["data"]
    for item in data_tmp:
        user_id = item["user_id"]
        if int(user_id.split("_")[2]) < N:
            data.append(item)

    with open(f"{root_path}/dataset/video_metadata_{VERSION}_new.json", "r") as json_file:
        video_metadata_tmp = json.load(json_file)
    video_metadata.update(video_metadata_tmp)

    with open(f"{root_path}/dataset/topic/tag2class_{VERSION}2.json", "r") as json_file:
        tag2class_tmp = json.load(json_file)
    tag2class.update(tag2class_tmp)

    with open(f"{root_path}/dataset/topic/class2id2.json", "r") as json_file:
        class2id_tmp = json.load(json_file)
    class2id.update(class2id_tmp)

video2class = {}
cnt = 0
cnt_fail = 0
S = 0
for video_id in video_metadata.keys():
    try:
        tags = video_metadata[video_id]["tags"].split(",")
        classes = {}
        classes_conf = {}
        
        for tag in tags:
            if tag == "":
                continue
            try:
                c = tag2class[tag][0]
                conf = tag2class[tag][1]
                if c not in classes.keys():
                    classes[c] = 0
                    classes_conf[c] = 0
                classes[c] += 1
                classes_conf[c] += conf
            except:
                continue
        
        video2class[video_id] = []

        max_conf = -10
        max_c = ""
        for c in classes.keys():
            classes_conf[c] /= classes[c]
            if max_conf < classes_conf[c]:
                max_c = c
                max_conf = classes_conf[c]
            if classes_conf[c] > 0:
                video2class[video_id].append(c)
        
    except:
        cnt_fail += 1
        continue

print(cnt_fail, len(video_metadata.keys()))

filtered_video_ids = {}
unique_home_video_id = {}

print(len(data))
# data_truth = data
for i in tqdm(range(len(data))):
    filtered_video_ids.update(dict(zip(data[i]["initial_homepage"], [1 for _ in range(len(data[i]["initial_homepage"]))])))

for i in tqdm(range(len(data))):
    video_views = data[i]["homepage"]
    tmp = {}
    # print(data[i]["viewed"])
    for video_view in video_views[:50]:
        for video_id in video_view:
            if video_id in filtered_video_ids.keys():
                continue
            if video_id not in unique_home_video_id.keys():
                unique_home_video_id[video_id] = 0
            if video_id not in tmp.keys():
                tmp[video_id] = 1
                unique_home_video_id[video_id] += 1
            else:
                tmp[video_id] = 1

removed_videos = []

for video in unique_home_video_id.keys():
    if unique_home_video_id[video] > 0 and unique_home_video_id[video] <= 20: # 18, real 31, reddit 19
        removed_videos.append(video)
for video in removed_videos:
    del unique_home_video_id[video]

filtered_video_ids.update(unique_home_video_id)

tmp = {}
for i in range(len(data)):
    if data[i]["user_id"] not in tmp.keys():
        tmp[data[i]["user_id"]] = []
    tmp[data[i]["user_id"]].append(data[i])
    
data = tmp

KL = not False
if not KL:
    # calculate the kl divergence
    def dist_metric(p, q):
        return sum([(p[i] - q[i]) ** 2 for i in range(len(p))])
else:
    # # calculate the kl divergence
    # def dist_metric(p, q):
    #     # return distance.jensenshannon(p, q, 2)
    #     return sum(p[i] * np.log2((p[i] + 1e-9)/(q[i] + 1e-9)) for i in range(len(p)))

    SENSITIVITY_W = [1 for _ in range(3)] + [154/27 for _ in range(3)] + [1 for _ in range(101)] + [154/27 for _ in range(24)] + [1 for _ in range(23)]
    def dist_metric(p, q):
        undesired_dist = 0
        desired_dist = 0
        t = 0
        for i in range(len(p)):
            if SENSITIVITY_W[i] > 1:
                undesired_dist += 1 * p[i] * np.log2((p[i] * 1 + 1e-9)/(q[i] * 1 + 1e-9))
                t += (q[i] > 0)
            else:
                desired_dist += SENSITIVITY_W[i] * p[i] * np.log2((p[i] + 1e-9)/(q[i] + 1e-9))
        return undesired_dist + desired_dist
        

def new_metric(p,q):
    res = 0
    for i in range(len(p)):
        res += abs(q[i] - p[i])
    return res


def get_cate_dist(data, filtered_video_ids):
    data_home_video_ids = {}
    last_cate_dict = {}
    last_label = [0 for _ in range(len(class2id))]

    for home_rec in data:
        for video_id in home_rec:
            if video_id in filtered_video_ids.keys():
                continue
            if video_id not in data_home_video_ids.keys():
                data_home_video_ids[video_id] = 0
            data_home_video_ids[video_id] += 1
            
    data_home_video_ids = {k : v for k, v in sorted(data_home_video_ids.items(), key=lambda item: item[1], reverse=True)}
    # print(len(data_home_video_ids))

    for video_id in list(data_home_video_ids.keys()):
        try:
            for cate in video2class[video_id]:
                if cate not in last_cate_dict.keys():
                    last_cate_dict[cate] = 0
                last_cate_dict[cate] += 1
        except:
            continue

    last_cate_dict = {k : v for k, v in sorted(last_cate_dict.items(), key=lambda item: item[1], reverse=True)}
    total_f = sum(list(last_cate_dict.values()))
    mean_f = 0 #np.mean(list(last_cate_dict.values()))
    std_f = 0 # np.std(list(last_cate_dict.values()))
    if not KL:
        # last_cate_dict = list(last_cate_dict.keys())[:20]
        for cate in list(last_cate_dict.keys()):
            if last_cate_dict[cate] >= 1 * mean_f + 1 * std_f:
                last_label[class2id[cate]] = 1
            elif last_cate_dict[cate] <= max(1 * mean_f - 1 * std_f, 0):
                last_label[class2id[cate]] = 0
            else:
                last_label[class2id[cate]] = 0
        # print(sum(last_label))
    else:
        for cate in last_cate_dict.keys():
            last_label[class2id[cate]] = last_cate_dict[cate] / total_f

    return last_label, data_home_video_ids

avg = 0
avg2 = 0
all_dist = 0

for i in tqdm(range(N)):
    base_cate_all = []
    avg_dist = 0
    avg_dist2 = 0
    dist = 0
    cnt = 0
    # for j in range(len(data[f"rl_base_{i}"])):
    #     rl_base_cate, rl_base_home_video_ids = get_cate_dist(data[f"rl_base_{i}"][j]["homepage"], filtered_video_ids)
    #     rand_base_cate, rand_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"][j]["homepage"], filtered_video_ids)
    #     # dist += dist_metric(rl_base_cate, rand_base_cate)
    #     # cnt += 1
    #     base_cate_all.append(rl_base_cate)
    # # dist /= len(data[f"rl_base_{i}"])
    # # all_dist += dist

    for j in range(len(data[f"rand_base_{i}"])):
        rand_base_cate, rand_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"][j]["homepage"], filtered_video_ids)
        base_cate_all.append(rand_base_cate)


    for k in range(4):
        for j in range(len(data[f"rand_base_{i}_{k+1}"])):
            rand_base_cate, rand_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}_{k+1}"][j]["homepage"], filtered_video_ids)
            base_cate_all.append(rand_base_cate)

    base_cate_all = np.array(base_cate_all)
    base_cate_all_mean = np.mean(base_cate_all, axis=0)
    # print(base_cate_all.shape)

    for k in range(len(base_cate_all)):
        cate_mean = np.zeros((154))
        cate_mean2 = np.zeros((154))
        for j in range(len(base_cate_all)):
            cate_mean += np.array(base_cate_all[j])
            
            if k != j:
                cate_mean2 += np.array(base_cate_all[j])
                dist += dist_metric(base_cate_all[j], base_cate_all[k])
                cnt += 1
        cate_mean /= (len(base_cate_all) )
        cate_mean2 /= (len(base_cate_all) - 1)
        avg_dist += dist_metric(cate_mean, base_cate_all[k])
        avg_dist2 += dist_metric(cate_mean2, base_cate_all[k])
    avg_dist /= len(base_cate_all)
    avg_dist2 /= len(base_cate_all)
    avg += avg_dist
    avg2 += avg_dist2
    all_dist += dist / cnt

print(avg / N, avg2 / N, (avg + avg2) / 2 / N, all_dist / N)

    


