import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

TAG = "final_new2_cate_test3"

PHASE = 0
with open(f"../dataset/category_ids_new.json", "r") as json_file:
    CATE_IDS = json.load(json_file)
del CATE_IDS[""]
del CATE_IDS["none"]

# calculate the kl divergence
def kl_divergence(p, q):
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

if PHASE == 0:
    VERSION = f"rl_{TAG}"
    with open(f"../dataset/sock_puppets_{VERSION}.json", "r") as json_file:
        data = json.load(json_file)[2]["data"]

    with open(f"../dataset/video_ids_{VERSION}.json", "r") as json_file:
        VIDEO_IDS = json.load(json_file)


    with open(f"../dataset/video_metadata_{VERSION}.json", "r") as json_file:
        VIDEO_METADATA = json.load(json_file)

    # with h5py.File(f"../dataset/video_embeddings_{VERSION}.hdf5", "r") as hf_emb:
    #     video_embeddings = hf_emb["embeddings"][:].astype("float32")



    print(len(data))

    initial_home_video_ids = {}
    unique_home_video_id = {}
    initial_home_video_ids_truth = {}
    unique_home_video_id_truth = {}

    AVG_R = 0
    with open(f"../dataset/sock_puppets_rand_final_new2_cate_test3.json", "r") as json_file:
        data_truth = json.load(json_file)[2]["data"]

    with open(f"../dataset/video_metadata_rand_final_new2_cate_test3.json", "r") as json_file:
        VIDEO_METADATA_TRUTH = json.load(json_file)


    # data_truth = data
    for i in tqdm(range(len(data))):
        initial_home_video_ids.update(dict(zip(data[i]["initial_homepage"], [1 for _ in range(len(data[i]["initial_homepage"]))])))
        video_views = data[i]["homepage"]
        tmp = {}
        # print(data[i]["viewed"])
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
        if unique_home_video_id[video] > 0 and unique_home_video_id[video] <= 30:
            removed_videos.append(video)
    for video in removed_videos:
        del unique_home_video_id[video]

    print(len(unique_home_video_id))
    # unique_home_video_id = {}
    # initial_home_video_ids = {}
    cate_dist = {}
    for i in tqdm(range(len(data_truth))):
        initial_home_video_ids_truth.update(dict(zip(data_truth[i]["initial_homepage"], [1 for _ in range(len(data_truth[i]["initial_homepage"]))])))
        video_views = data_truth[i]["homepage"]
        tmp = {}
        # print(data_truth[i]["viewed"])
        for video_view in video_views:
            for video_id in video_view:
                if video_id in initial_home_video_ids_truth.keys():
                    continue
                if video_id not in unique_home_video_id_truth.keys():
                    unique_home_video_id_truth[video_id] = 0
                if video_id not in tmp.keys():
                    tmp[video_id] = 1
                    unique_home_video_id_truth[video_id] += 1
                else:
                    tmp[video_id] = 1
    print(len(initial_home_video_ids_truth))
    removed_videos = []
    for video in unique_home_video_id_truth.keys():
        if unique_home_video_id_truth[video] > 0 and unique_home_video_id_truth[video] <= 30:
            removed_videos.append(video)
    for video in removed_videos:
        del unique_home_video_id_truth[video]

    print(len(unique_home_video_id_truth))
    # unique_home_video_id_truth = {}



    avergea_r = 0
    N = 1500
    k = 1
    cnt = 0
    similarity = 0

    for i in range(N):
        base_data = data[i]["homepage"]
        base_home_video_ids = {}
        base_cate = [0.01 for _ in range(len(CATE_IDS))]
        for home_rec in base_data:
            for video_id in home_rec:
                if video_id in initial_home_video_ids.keys():
                    continue
                if video_id in unique_home_video_id.keys():
                    continue
                if video_id not in base_home_video_ids.keys():
                    base_home_video_ids[video_id] = 0
                base_home_video_ids[video_id] += 1
                # if "categories" in VIDEO_METADATA[video_id].keys():
                #     try:
                #         base_cate[CATE_IDS[VIDEO_METADATA[video_id]["categories"]]] += 1
                #     except:
                #         continue
        base_home_video_ids = {k : v for k, v in sorted(base_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
        base_home_video_ids_list = list(base_home_video_ids.keys())
        for video_id in base_home_video_ids_list:
            if "categories" in VIDEO_METADATA[video_id].keys():
                try:
                    base_cate[CATE_IDS[VIDEO_METADATA[video_id]["categories"]]] += 1
                except:
                    continue
        # base_emb = np.mean(video_embeddings[base_home_video_ids_list], axis=0)
        # print(base_cate, len(base_cate))

        obfu_data = data[i+N]["homepage"]
        obfu_home_video_ids = {}
        obfu_cate = [0.01 for _ in range(len(CATE_IDS))]
        for home_rec in obfu_data:
            for video_id in home_rec:
                if video_id in initial_home_video_ids.keys():
                    continue
                if video_id in unique_home_video_id.keys():
                    continue
                if video_id not in obfu_home_video_ids.keys():
                    obfu_home_video_ids[video_id] = 0
                obfu_home_video_ids[video_id] += 1
                # if "categories" in VIDEO_METADATA[video_id].keys():
                #     try:
                #         obfu_cate[CATE_IDS[VIDEO_METADATA[video_id]["categories"]]] += 1
                #     except:
                #         continue
        obfu_home_video_ids = {k : v for k, v in sorted(obfu_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
        obfu_home_video_ids_list = list(obfu_home_video_ids.keys())
        for video_id in obfu_home_video_ids_list:
            if "categories" in VIDEO_METADATA[video_id].keys():
                try:
                    obfu_cate[CATE_IDS[VIDEO_METADATA[video_id]["categories"]]] += 1
                except:
                    continue
        # obfu_emb = video_embeddings[obfu_home_video_ids_list]
        # obfu_emb = np.mean(video_embeddings[base_home_video_ids_list], axis=0)
        # print(obfu_cate, len(obfu_cate))
        # base_cate = [int(i) // 10 + 1 for i in base_cate]
        # obfu_cate = [int(i) // 10 + 1 for i in obfu_cate]
        base_cate = [base_cate[i] / sum(base_cate) for i in range(len(base_cate))]
        obfu_cate = [obfu_cate[i] / sum(obfu_cate) for i in range(len(base_cate))]
        # print(i, base_cate)
        # print(i, obfu_cate)
        # delta = [(base_cate[i] - obfu_cate[i]) ** 2 for i in range(len(base_cate))]
        # print(np.mean(delta))
        # AVG_R += kl_divergence(base_cate, obfu_cate)

        # similarity += np.sum(base_emb * obfu_emb)

        delta_len = len(data[i+N]["viewed"]) - len(data[i]["viewed"])

        # print(sorted(base_home_video_ids))
        # print(sorted(obfu_home_video_ids))
        # seed_video = data[i]["viewed"][0]
        item_cate = [0.01 for _ in range(len(CATE_IDS))]
        for item in data_truth:
            # if item["seedId"] == seed_video and item["viewed"][1] == data[i]["viewed"][1]:
            item = data_truth[i]
            if item["viewed"][0] == data[i]["viewed"][0] and item["viewed"][1] == data[i]["viewed"][1]:
                item_home_video_ids = {}
                for home_rec in item["homepage"]:
                    for video_id in home_rec:
                        if video_id in initial_home_video_ids_truth.keys():
                            continue
                        if video_id in unique_home_video_id_truth.keys():
                            continue
                        if video_id not in item_home_video_ids.keys():
                            item_home_video_ids[video_id] = 0
                        item_home_video_ids[video_id] += 1
                        # if "categories" in VIDEO_METADATA_TRUTH[video_id].keys():
                        #     try:
                        #         item_cate[CATE_IDS[VIDEO_METADATA_TRUTH[video_id]["categories"]]] += 1
                        #     except:
                        #         continue
                item_home_video_ids = {k : v for k, v in sorted(item_home_video_ids.items(), key=lambda item: item[1], reverse=True)[0:100]}
                break
                # print(sorted(item_home_video_ids))
        for video_id in item_home_video_ids.keys():
            if "categories" in VIDEO_METADATA_TRUTH[video_id].keys():
                try:
                    item_cate[CATE_IDS[VIDEO_METADATA_TRUTH[video_id]["categories"]]] += 1
                except:
                    continue
        # break
        
        item_cate = [item_cate[i] / sum(item_cate) for i in range(len(item_cate))]
        # print(i, item_cate)
        if delta_len > 6:
            # AVG_R += np.sqrt(np.sum((np.array(base_cate) - np.array(item_cate)) ** 2)) # kl_divergence(base_cate, item_cate)
            AVG_R += kl_divergence(base_cate, item_cate)
            cnt += 1
            # fig, ax = plt.subplots()
            # plt.plot(base_cate)
            # plt.plot(obfu_cate)
            # plt.legend(["non-obfu","obfu"])
            # plt.xticks(np.arange(0, 16, 1.0))
            # ax.set_xticklabels(CATE_IDS.keys())
            # plt.xticks(rotation=-90)
            # plt.savefig(f"./figs/{VERSION}_sample_{i}.png", bbox_inches='tight')
            
            cate_dist[i] = {"non-obfu": base_cate, "obfu": obfu_cate}
            removed = 0
            added = 0
            
            # if i > 50:
            #     print(i, item["homepage"])
            #     # print(i, data[i+N]["viewed"])
            #     print(i, data[i]["homepage"])

            for video in base_home_video_ids.keys():
                if video not in item_home_video_ids.keys():
                    removed += 1
            for video in item_home_video_ids.keys():
                if video not in base_home_video_ids.keys():
                    added += 1
            avergea_r += (removed + added)

    print(avergea_r / cnt, similarity / N, AVG_R / cnt)

    with open(f"./figs/{VERSION}_metadata.json","w") as json_file:
        json.dump(cate_dist, json_file)

else:
    VERSION = f"rl_{TAG}"
    with open(f"./figs/{VERSION}_metadata.json","r") as json_file:
        cate_dist_rl = json.load(json_file)

    VERSION = f"rand_{TAG}"
    with open(f"./figs/{VERSION}_metadata.json","r") as json_file:
        cate_dist_rand = json.load(json_file)

    for i in cate_dist_rl.keys():
        kl = []
        base_cate_rl = cate_dist_rl[i]["non-obfu"]
        obfu_cate_rl = cate_dist_rl[i]["obfu"]
        base_cate_rand = cate_dist_rand[i]["non-obfu"]
        obfu_cate_rand = cate_dist_rand[i]["obfu"]
        fig, ax = plt.subplots()
        plt.plot(base_cate_rl,"*-")
        plt.plot(obfu_cate_rl,"o-")
        plt.legend(["rl-non-obfu","rl-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence caused by rl obfuscator: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40 \n 20% obfuscation URLs".format(
            kl_divergence(base_cate_rl, obfu_cate_rl),
            np.sqrt(np.sum((np.array(base_cate_rl) - np.array(obfu_cate_rl)) ** 2))
            )
        )
        plt.savefig(f"./figs/rl_sample_{i}.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.plot(base_cate_rand,"*-")
        plt.plot(obfu_cate_rand,"o-")
        plt.legend(["rand-non-obfu","rand-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence caused by  rand obfuscator: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40 \n 20% obfuscation URLs".format(
            kl_divergence(base_cate_rand, obfu_cate_rand),
            np.sqrt(np.sum((np.array(base_cate_rand) - np.array(obfu_cate_rand)) ** 2))
            )
        )
        plt.savefig(f"./figs/rand_sample_{i}.png", bbox_inches='tight')

        fig, ax = plt.subplots()
        plt.plot(base_cate_rl,"*-")
        plt.plot(base_cate_rand,"*-")
        plt.legend(["rl-non-obfu","rand-non-obfu"])
        plt.xticks(np.arange(0, 16, 1.0))
        ax.set_xticklabels(CATE_IDS.keys())
        plt.xticks(rotation=-90)
        plt.title("KL divergence: {:.4f}\n L2 distance: {:.4f}\n non-obfuscated persona length: 40".format(
            kl_divergence(base_cate_rl, base_cate_rand),
            np.sqrt(np.sum((np.array(base_cate_rand) - np.array(base_cate_rl)) ** 2))
            )
        )
        plt.savefig(f"./figs/base_sample_{i}.png", bbox_inches='tight')
