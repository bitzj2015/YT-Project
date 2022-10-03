import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(0)
root_path = "/scratch/YT_dataset"

VERSION = "final_joint_cate_100_2_test"
# VERSION = "final_joint_cate_100_2_0.1"
VERSION = "final_joint_cate_103_2_test"
VERSION = "reddit_cate_100_2_test"
VERSION = "latest_joint_cate_010"
VERSION = "latest_joint_cate_010_reddit3_0.2"
VERSION = "latest_joint_cate_010_0.3"
VERSION = "realuser_0.2_test"

PHASE = 0
with open(f"{root_path}/dataset/category_ids_latest.json", "r") as json_file:
    CATE_IDS = json.load(json_file)
print(len(CATE_IDS))

with open(f"{root_path}/dataset/sock_puppets_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]

with open(f"{root_path}/dataset/video_ids_{VERSION}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)

with open(f"{root_path}/dataset/video_metadata_{VERSION}.json", "r") as json_file:
    VIDEO_METADATA = json.load(json_file)


# calculate the kl divergence
def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


def new_metric(p,q):
    res = 0
    for i in range(len(p)):
        res += abs(q[i] - p[i])
    return res


def get_cate_dist(data, filtered_video_ids):
    data_home_video_ids = {}
    data_home_video_tags = {}
    data_cate = [0.001 for _ in range(len(CATE_IDS))]
    for home_rec in data:
        for video_id in home_rec:
            if video_id in filtered_video_ids.keys():
                continue
            if video_id not in data_home_video_ids.keys():
                data_home_video_ids[video_id] = 0
            data_home_video_ids[video_id] += 1
            if "tags" in VIDEO_METADATA[video_id].keys():
                for tag in VIDEO_METADATA[video_id]["tags"].split(","):
                    tag = tag.lower()
                    if tag not in data_home_video_tags.keys():
                        data_home_video_tags[tag] = 0
                    data_home_video_tags[tag] += 1
            
    data_home_video_ids = {k : v for k, v in sorted(data_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
    data_home_video_ids_list = list(data_home_video_ids.keys())
    data_home_video_tags = {k : v for k, v in sorted(data_home_video_tags.items(), key=lambda item: item[1], reverse=True)[0:100]}
    data_home_video_tags_list = list(data_home_video_tags.keys())
    # print(data_home_video_ids)
    for video_id in data_home_video_ids_list:
        if "categories" in VIDEO_METADATA[video_id].keys():
            try:
                data_cate[CATE_IDS[VIDEO_METADATA[video_id]["categories"]]] += 1 # data_home_video_ids[video_id]
            except:
                continue

    data_cate = [data_cate[i] / sum(data_cate) for i in range(len(data_cate))]
    return data_cate, data_home_video_tags


if PHASE == 0:
    print(len(data))
    filtered_video_ids = {}
    unique_home_video_id = {}

    # data_truth = data
    for i in tqdm(range(len(data))):
        filtered_video_ids.update(dict(zip(data[i]["initial_homepage"], [1 for _ in range(len(data[i]["initial_homepage"]))])))

    for i in tqdm(range(len(data))):
        video_views = data[i]["homepage"]
        tmp = {}
        # print(data[i]["viewed"])
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
    print(len(filtered_video_ids))
    for video in unique_home_video_id.keys():
        if unique_home_video_id[video] > 0 and unique_home_video_id[video] <= 1000:
            removed_videos.append(video)
    for video in removed_videos:
        del unique_home_video_id[video]

    filtered_video_ids.update(unique_home_video_id)
    print(len(filtered_video_ids))

    avergea_r = 0
    N = 200
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

    save_data = {}
    T = 0
    for i in tqdm(range(N)):
        # if i < 30:
        #     continue
        try:
            rl_base_cate, rl_base_home_video_ids = get_cate_dist(data[f"rl_base_{i}"]["homepage"], filtered_video_ids)
            rl_obfu_cate, rl_obfu_home_video_ids = get_cate_dist(data[f"rl_obfu_{i}"]["homepage"], filtered_video_ids)
            bias_obfu_cate, bias_obfu_home_video_ids = get_cate_dist(data[f"bias_obfu_{i}"]["homepage"], filtered_video_ids)
            rand_base_cate, rand_base_home_video_ids = get_cate_dist(data[f"rand_base_{i}"]["homepage"], filtered_video_ids)
            rand_base_cate2, _ = get_cate_dist(data[f"rand_base_{random.randint(30,N-1)}"]["homepage"], filtered_video_ids)
            rand_obfu_cate, rand_obfu_home_video_ids = get_cate_dist(data[f"rand_obfu_{i}"]["homepage"], filtered_video_ids)

            rand_base_cate_view, _ = get_cate_dist([data[f"rand_base_{i}"]["viewed"]], {})
            rand_avg_kl_view += kl_divergence(rand_base_cate, rand_base_cate_view)
            # print(rand_base_cate, rand_base_cate_view)
  
            save_data[f"rl_base_{i}"] = data[f"rl_base_{i}"]
            save_data[f"rl_obfu_{i}"] = data[f"rl_obfu_{i}"]
            save_data[f"bias_obfu_{i}"] = data[f"bias_obfu_{i}"]
            save_data[f"rand_base_{i}"] = data[f"rand_base_{i}"]
            save_data[f"rand_obfu_{i}"] = data[f"rand_obfu_{i}"]

            save_data[f"rl_base_{i}"]["cate_dist"] = rl_base_cate
            save_data[f"rl_obfu_{i}"]["cate_dist"] = rl_obfu_cate
            save_data[f"bias_obfu_{i}"]["cate_dist"] = bias_obfu_cate
            save_data[f"rand_base_{i}"]["cate_dist"] = rand_base_cate
            save_data[f"rand_obfu_{i}"]["cate_dist"] = rand_obfu_cate

            delta_len = len(data[f"rl_obfu_{i}"]["viewed"]) - len(data[f"rl_base_{i}"]["viewed"])
            
            if delta_len > 0:
                T += delta_len
                rl_avg_kl += kl_divergence(rand_base_cate, rl_obfu_cate)
                bias_avg_kl += kl_divergence(rand_base_cate, bias_obfu_cate)
                rand_avg_kl += kl_divergence(rand_base_cate, rand_obfu_cate)
                base_avg_kl += kl_divergence(rand_base_cate, rand_base_cate2)

                rl_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rl_obfu_cate)) ** 2))
                bias_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(bias_obfu_cate)) ** 2))
                rand_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rand_obfu_cate)) ** 2))
                base_avg_l2 += np.sqrt(np.sum((np.array(rand_base_cate) - np.array(rand_base_cate2)) ** 2))

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
                # rl_avg_dist += added / len(rl_obfu_home_video_ids)
                # print(removed, added)
                # print(data[f"rl_obfu_{i}"]["viewed"])
                # print(rl_obfu_home_video_ids)
                # print(rand_base_home_video_ids)
                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in bias_obfu_home_video_ids.keys():
                        removed += 1
                for video in bias_obfu_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                bias_avg_dist += (removed + added) / len(rand_base_home_video_ids)
                # bias_avg_dist += added / len(bias_obfu_home_video_ids)
                # print(removed, added)
                # print(data[f"bias_obfu_{i}"]["viewed"])
                # print(bias_obfu_home_video_ids)
                # print(rand_base_home_video_ids)
                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in rand_obfu_home_video_ids.keys():
                        removed += 1
                for video in rand_obfu_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                rand_avg_dist += (removed + added) / len(rand_base_home_video_ids)
                # print(removed, added)
                # print(data[f"rand_obfu_{i}"]["viewed"])
                # print(rand_obfu_home_video_ids)
                # print(rand_base_home_video_ids)
                removed = 0
                added = 0

                for video in rand_base_home_video_ids.keys():
                    if video not in rl_base_home_video_ids.keys():
                        removed += 1
                for video in rl_base_home_video_ids.keys():
                    if video not in rand_base_home_video_ids.keys():
                        added += 1
                base_avg_dist += (removed + added) / len(rand_base_home_video_ids)
                # base_avg_dist += added / 100 # len(rl_base_home_video_ids.keys())

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

    with open(f"./figs/dataset_{VERSION}.json","w") as json_file:
        json.dump(save_data, json_file)

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
            kl_divergence(rl_base_cate_rl, rl_obfu_cate_rl),
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
            kl_divergence(rl_base_cate_rand, rl_obfu_cate_rand),
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
            kl_divergence(rl_base_cate_rl, rl_base_cate_rand),
            np.sqrt(np.sum((np.array(rl_base_cate_rand) - np.array(rl_base_cate_rl)) ** 2))
            )
        )
        plt.savefig(f"./figs/rl_base_sample_{i}.png", bbox_inches='tight')
