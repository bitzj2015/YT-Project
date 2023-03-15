import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from copy import deepcopy
random.seed(0)
root_path = "/scratch/YT_dataset"

# VERSION = "final_joint_cate_100_2_test"
# VERSION = "final_joint_cate_100_2_0.1"
# VERSION = "final_joint_cate_103_2_test"
# VERSION = "reddit_cate_100_2_test"
# VERSION = "latest_joint_cate_010"
# VERSION = "latest_joint_cate_010_reddit3_0.2"
# VERSION = "latest_joint_cate_010_0.3"
# VERSION = "v1_binary_0.2_test"
# VERSION = "0.3_v2_kldiv_0.3_test_0.2_test"
# VERSION = "realuser_0.2_test"
# VERSION = "0.2_v2_kldiv_0.2_test_feb"
VERSION = "0.2_v2_kldiv_0.2_test_feb"
# VERSION = "realuser"
# VERSION = "0.2_v2_kldiv_reddit_test"
# VERSION = "0.2_v2_kldiv_reddit2_test"
# VERSION = "v2_kldiv_sensitive"
# VERSION = "realuser_all"
VERSION2 = "0.2_v2_kldiv_pbooster_3_new"
VERSION2 = "0.2_v2_kldiv_pbooster_reddit_0.2_3_new_v2"
VERSION2 = "0.7_v2_kldiv_feb_0.7_0_new_v2"
# VERSION2 = "0.2_realuser_pbooster_3_new_v2"
# VERSION2 = "0.5_realuser_0_new_v2"

stat = {}
N = 1000
TH = 35 # new reddit 35
PHASE = 0
with open(f"{root_path}/dataset/category_ids_latest.json", "r") as json_file:
    CATE_IDS = json.load(json_file)

with open(f"{root_path}/dataset/sock_puppets_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]

# with open(f"{root_path}/dataset/sock_puppets_0.3_v2_kldiv_0.3_test.json", "r") as json_file:
#     data_base = json.load(json_file)[2]["data"]

with open(f"{root_path}/dataset/sock_puppets_{VERSION2}.json", "r") as json_file:
    data_base = json.load(json_file)[2]["data"]

data = []
for item in data_base:
    # if item["user_id"].startswith("rand_base"):
    #     item["user_id"] = item["user_id"].replace("rand_base", "rl_base")
    #     data.append(deepcopy(item))

    if item["user_id"].startswith("bias_base"):
        data.append(deepcopy(item))
        item["user_id"] = item["user_id"].replace("bias_base", "pb_base")
        data.append(deepcopy(item))
        item["user_id"] = item["user_id"].replace("pb_base", "rand_base")
        data.append(deepcopy(item))

    if item["user_id"].startswith("bias_obfu"):
        data.append(deepcopy(item))
        item["user_id"] = item["user_id"].replace("bias_obfu", "pb_obfu")
        data.append(deepcopy(item))
        item["user_id"] = item["user_id"].replace("pb_obfu", "rand_obfu")
        data.append(deepcopy(item))

# with open(f"{root_path}/dataset/video_metadata_0.3_v2_kldiv_0.3_test_new.json", "r") as json_file:
#     VIDEO_METADATA = json.load(json_file)

with open(f"{root_path}/dataset/video_metadata_{VERSION2}_new.json", "r") as json_file:
    VIDEO_METADATA = json.load(json_file)

with open(f"{root_path}/dataset/video_metadata_{VERSION}_new.json", "r") as json_file:
    VIDEO_METADATA.update(json.load(json_file))

with open(f"{root_path}/dataset/topic/tag2class_{VERSION}2.json", "r") as json_file:
    tag2class = json.load(json_file)

with open(f"{root_path}/dataset/topic/tag2class_{VERSION2}2.json", "r") as json_file:
    tag2class.update(json.load(json_file))

with open(f"{root_path}/dataset/topic/class2id2.json", "r") as json_file:
    class2id = json.load(json_file)

id2class = {}
for cate in class2id.keys():
    id2class[class2id[cate]] = cate
    stat[cate] = 0.1

avg_stat = {"avg_num_class":0, "avg_num_class_p":0}
video2class = {}
cnt = 0
cnt_fail = 0
S = 0
for video_id in VIDEO_METADATA.keys():
    classes = {}
    classes_conf = {}
    try:
        tags = VIDEO_METADATA[video_id]["tags"].split(",")

        
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
            if classes_conf[c] > 0:
                video2class[video_id].append(c)
        
    except:
        cnt_fail += 1
        continue

print(cnt_fail, len(VIDEO_METADATA.keys()))

KL = not False
if not KL:
    # calculate the kl divergence
    def dist_metric(p, q):
        return sum([(p[i] - q[i]) ** 2 for i in range(len(p))])
else:
    # calculate the kl divergence
    def dist_metric(p, q):
        # return distance.jensenshannon(p, q, 2)
        return sum(p[i] * np.log2((p[i] + 1e-9)/(q[i] + 1e-9)) for i in range(len(p)))

    # SENSITIVITY_W = [1 for _ in range(3)] + [154/27 for _ in range(3)] + [1 for _ in range(101)] + [154/27 for _ in range(24)] + [1 for _ in range(23)]
    # def dist_metric(p, q):
    #     undesired_dist = 0
    #     desired_dist = 0
    #     t = 0
    #     for i in range(len(p)):
    #         if SENSITIVITY_W[i] > 1:
    #             undesired_dist += p[i] * np.log2((p[i] * 1 + 1e-9)/(q[i] * 1 + 1e-9))
    #             t += q[i] > 1e-4
    #         else:
    #             desired_dist += SENSITIVITY_W[i] * p[i] * np.log2((p[i] + 1e-9)/(q[i] + 1e-9))
    #     return undesired_dist + desired_dist
        

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
    mean_f = 0 # np.mean(list(last_cate_dict.values()))
    std_f = 0 #np.std(list(last_cate_dict.values()))
    if not KL:
        # last_cate_dict = list(last_cate_dict.keys())[:20]
        for cate in list(last_cate_dict.keys()):
            if last_cate_dict[cate] >= 1 * mean_f + 1 * std_f:
                last_label[class2id[cate]] = 1
            elif last_cate_dict[cate] <= max(1 * mean_f - 1 * std_f, 0):
                last_label[class2id[cate]] = 0
            else:
                last_label[class2id[cate]] = 0

    else:
        for cate in last_cate_dict.keys():
            last_label[class2id[cate]] = last_cate_dict[cate] / total_f

    return last_label, data_home_video_ids


if PHASE == 0:
    filtered_video_ids = {}
    unique_home_video_id = {}

    for i in tqdm(range(len(data))):
        filtered_video_ids.update(dict(zip(data[i]["initial_homepage"], [1 for _ in range(len(data[i]["initial_homepage"]))])))

    for i in tqdm(range(len(data))):
        video_views = data[i]["homepage"]
        tmp = {}

        for video_view in video_views:
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
        if unique_home_video_id[video] > 0 and unique_home_video_id[video] <= TH: # 52, 54, 44 # realuser 61 # reddit 75
            removed_videos.append(video)
    for video in removed_videos:
        del unique_home_video_id[video]

    filtered_video_ids.update(unique_home_video_id)

    avergea_r = 0
    
    k = 1
    cnt = 0
    miss_cnt = 0
    similarity = 0
    rl_avg_kl = 0
    rl_avg_l2 = 0
    rl_avg_dist = 0

    bias_avg_kl = 0
    bias_avg_l2 = 0
    bias_avg_dist = 0

    rand_avg_kl = 0
    rand_avg_l2 = 0
    rand_avg_dist = 0
    rand_avg_kl_view = 0

    base_avg_kl = 0
    base_avg_l2 = 0
    base_avg_dist = 0

    cate_dist = {}

    tmp = {}
    for i in range(len(data)):
        tmp[data[i]["user_id"]] = data[i]
        
    data = tmp

    # avg_obf = []
    # for i in tqdm(range(N)):
    #     rl_obfu_cate, rl_obfu_home_video_ids = get_cate_dist(data[f"rl_obfu_{i}"]["homepage"], filtered_video_ids)
    #     bias_obfu_cate, bias_obfu_home_video_ids = get_cate_dist(data[f"bias_obfu_{i}"]["homepage"], filtered_video_ids)
    #     rand_obfu_cate, rand_obfu_home_video_ids = get_cate_dist(data[f"rand_obfu_{i}"]["homepage"], filtered_video_ids)
    #     avg_obf.append(rl_obfu_cate)
    #     avg_obf.append(bias_obfu_cate)
    #     avg_obf.append(rand_obfu_cate)

    
    # avg_obf = np.array(avg_obf)
    # avg_obf = np.mean(avg_obf, axis=0)

    save_data = {}
    T = 0
    for i in tqdm(range(N)):
        # if i < 30:
        #     continue
        try:
            if len(data[f"pb_obfu_{i}"]["viewed"]) <  100:
                continue
            rl_base_cate, rl_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"]["homepage"], filtered_video_ids)
            rl_obfu_cate, rl_obfu_home_video_ids = get_cate_dist(data[f"pb_obfu_{i}"]["homepage"], filtered_video_ids)
            bias_base_cate, bias_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"]["homepage"], filtered_video_ids)
            bias_obfu_cate, bias_obfu_home_video_ids = get_cate_dist(data[f"bias_obfu_{i}"]["homepage"], filtered_video_ids)
            rand_base_cate, rand_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"]["homepage"], filtered_video_ids)
            rand_base_cate2, _ = get_cate_dist(data[f"rand_base_{random.randint(0,N-1)}"]["homepage"], filtered_video_ids)
            rand_obfu_cate, rand_obfu_home_video_ids = get_cate_dist(data[f"rand_obfu_{i}"]["homepage"], filtered_video_ids)
            S += sum(rl_obfu_cate)

            rand_base_cate_view, _ = get_cate_dist([data[f"rand_base_{i}"]["viewed"]], {})
            class_id = np.argsort(rand_base_cate_view)[-15:]
            avg_stat["avg_num_class"] += (np.array(rand_base_cate_view) > 0).sum()
            
            for idx in class_id:
                stat[id2class[idx]] += 1
                avg_stat["avg_num_class_p"] += rand_base_cate_view[idx]
            
            
            # if i < 5:
            #     print(data[f"rand_base_{i}"]["viewed"])
            #     print(data[f"rl_base_{i}"]["viewed"])
            #     for video  in rand_base_home_video_ids:
            #         if "tags" in VIDEO_METADATA[video].keys():
            #             print(VIDEO_METADATA[video]["tags"])
            #             print(video2class[video], VIDEO_METADATA[video]["title"], video)

            rand_avg_kl_view += dist_metric(rand_base_cate, rand_base_cate_view)

            save_data[f"rl_base_{i}"] = data[f"rand_base_{i}"]
            save_data[f"rl_obfu_{i}"] = data[f"pb_obfu_{i}"]
            save_data[f"pb_base_{i}"] = data[f"pb_base_{i}"]
            save_data[f"pb_obfu_{i}"] = data[f"pb_obfu_{i}"]
            save_data[f"rand_base_{i}"] = data[f"rand_base_{i}"]
            save_data[f"rand_obfu_{i}"] = data[f"rand_obfu_{i}"]

            save_data[f"rl_base_{i}"]["cate_dist"] = rl_base_cate
            save_data[f"rl_obfu_{i}"]["cate_dist"] = rl_obfu_cate
            save_data[f"pb_base_{i}"]["cate_dist"] = bias_base_cate
            save_data[f"pb_obfu_{i}"]["cate_dist"] = bias_obfu_cate
            save_data[f"rand_base_{i}"]["cate_dist"] = rand_base_cate
            save_data[f"rand_obfu_{i}"]["cate_dist"] = rand_obfu_cate

            delta_len = len(data[f"rand_obfu_{i}"]["viewed"]) - len(data[f"rand_base_{i}"]["viewed"])
            
            if delta_len > 0:
                T += delta_len
                # rand_base_cate = list(avg_obf)
                rl_avg_kl += dist_metric(rl_obfu_cate, rl_base_cate)
                bias_avg_kl += dist_metric(bias_obfu_cate, rand_base_cate)
                rand_avg_kl += dist_metric(rand_obfu_cate, rand_base_cate)
                base_avg_kl += dist_metric(rand_base_cate2, rand_base_cate)

                rl_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rl_obfu_cate)) ** 2))
                bias_avg_l2 += np.sqrt(np.sum((np.array(bias_base_cate) - np.array(bias_obfu_cate)) ** 2))
                rand_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rand_obfu_cate)) ** 2))
                base_avg_l2 += dist_metric((np.array(rl_base_cate) + np.array(rl_base_cate) * 1) / 2, rand_base_cate) # np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rl_base_cate)) ** 2))

                cnt += 1

                cate_dist[i] = {
                    "rl_base": rl_base_cate, "rl_obfu": rl_obfu_cate, 
                    "bias_base": rand_base_cate, "bias_obfu": bias_obfu_cate, 
                    "rand_base": rand_base_cate, "rand_obfu": rand_obfu_cate
                }
                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in rl_obfu_home_video_ids.keys():
                        removed += 1
                for video in rl_obfu_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                rl_avg_dist += (removed + added) / len(rand_base_home_video_ids)

                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in bias_obfu_home_video_ids.keys():
                        removed += 1
                for video in bias_obfu_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                bias_avg_dist += (removed + added) / len(rand_base_home_video_ids)

                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in rand_obfu_home_video_ids.keys():
                        removed += 1
                for video in rand_obfu_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                rand_avg_dist += (removed + added) / len(rand_base_home_video_ids)

                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in rl_base_home_video_ids.keys():
                        removed += 1
                for video in rl_base_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                base_avg_dist += (removed + added) / len(rand_base_home_video_ids)

        except:
            miss_cnt += 1
            continue

    print(miss_cnt, cnt, T/cnt)
    print("[base] KL: {}, L2: {}, Videos: {}".format(base_avg_kl / cnt, base_avg_l2 / cnt, base_avg_dist / cnt))
    print("[rand] KL: {}, L2: {}, Videos: {}".format(rand_avg_kl / cnt, rand_avg_l2 / cnt, rand_avg_dist / cnt))
    print("[bias] KL: {}, L2: {}, Videos: {}".format(bias_avg_kl / cnt, bias_avg_l2 / cnt, bias_avg_dist / cnt))
    print("[rl] KL: {}, L2: {}, Videos: {}".format(rl_avg_kl / cnt, rl_avg_l2 / cnt, rl_avg_dist / cnt))
    print("[baseline-U] KL: {}".format(rand_avg_kl_view / cnt))

    with open(f"./figs/{VERSION}_metadata.json","w") as json_file:
        json.dump(cate_dist, json_file)

    with open(f"./figs/dataset_{VERSION2}.json","w") as json_file:
        json.dump(save_data, json_file)
    
    print(S / N)

else:
    with open(f"./figs/{VERSION}_metadata.json","r") as json_file:
        cate_dist_rl = json.load(json_file)

    with open(f"./figs/{VERSION}_metadata.json","r") as json_file:
        cate_dist_rand = json.load(json_file)

    for i in cate_dist_rl.keys():
        kl = []
        rl_base_cate_rl = cate_dist_rl[i]["non-obfu"]
        rl_obfu_cate_rl = cate_dist_rl[i]["obfu"]
        rl_base_cate_rand = cate_dist_rand[i]["non-obfu"]
        rl_obfu_cate_rand = cate_dist_rand[i]["obfu"]
        fig, ax = plt.subplots()
        plt.plot(rl_base_cate_rl,"*-")
        plt.plot(rl_obfu_cate_rl,"o-")
        plt.legend(["rl-non-obfu","rl-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence caused by rl obfuscator: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40 \n 20% obfuscation URLs".format(
            dist_metric(rl_base_cate_rl, rl_obfu_cate_rl),
            np.sqrt(np.sum((np.array(rl_base_cate_rl) - np.array(rl_obfu_cate_rl)) ** 2))
            )
        )
        plt.savefig(f"./figs/rl_sample_{i}.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.plot(rl_base_cate_rand,"*-")
        plt.plot(rl_obfu_cate_rand,"o-")
        plt.legend(["rand-non-obfu","rand-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence caused by  rand obfuscator: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40 \n 20% obfuscation URLs".format(
            dist_metric(rl_base_cate_rand, rl_obfu_cate_rand),
            np.sqrt(np.sum((np.array(rl_base_cate_rand) - np.array(rl_obfu_cate_rand)) ** 2))
            )
        )
        plt.savefig(f"./figs/rand_sample_{i}.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.plot(rl_base_cate_rl,"*-")
        plt.plot(rl_base_cate_rand,"*-")
        plt.legend(["rl-non-obfu","rand-non-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40".format(
            dist_metric(rl_base_cate_rl, rl_base_cate_rand),
            np.sqrt(np.sum((np.array(rl_base_cate_rand) - np.array(rl_base_cate_rl)) ** 2))
            )
        )
        plt.savefig(f"./figs/rl_base_sample_{i}.png", bbox_inches='tight')


import matplotlib.pyplot as plt

plt.figure()
plt.bar([i for i in range(len(stat))], np.array(list(stat.values()))/936)
plt.xlabel("Video class index")
plt.ylabel("No. of users who view it frequently")
plt.savefig("stat.png")

plt.figure()
plt.bar([i for i in range(len(stat))], np.log10(list(stat.values())))
plt.xlabel("Video class index")
plt.ylabel("No. of users who view it frequently ($log_{10}$)")
plt.savefig("stat2.png")
# print(avg_stat["avg_num_class"] / N, avg_stat["avg_num_class_p"] / N)
# print(stat)