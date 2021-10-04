from sentence_transformers import SentenceTransformer, util
import json
import h5py
import random
import numpy as np
from tqdm import tqdm

model = SentenceTransformer('all-mpnet-base-v2')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

query_embedding = model.encode("What started as a simple desire for a brighter plasma globe got carried away and I ended up building a million volt lightning tower of death..")
passage_embedding = model.encode(["Aqui te traigo la esperada película de Space Jam 2, que habla acerca de  l lazo entre padre e hijo.",
                                  "Car Shortage PKG", "Credit goes to Tyler1.","These 3 games are really cool and fun, here are the links if you want more info","In Monday's stock market breakdown"])

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
print(query_embedding.shape, passage_embedding.shape)


with open("../dataset/id_videos_new.json", "r") as file:
    id_videos = json.load(file)
with open("../dataset/video_adj_list_new_w.json", "r") as file:
    video_adj_list = json.load(file)
with h5py.File("../dataset/video_embeddings_new.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:]
with h5py.File("../dataset/video_embeddings_new_des.hdf5", "r") as hf_emb:
    video_embeddings_des = hf_emb["embeddings"][:]
with open("../dataset/video2category_ids_new.json", "r") as file:
    video2category_ids = json.load(file)

debug = 0

def run_related_video_analysis(id_videos, video_adj_list, embeddings):
    count = 0
    gain_mean = 0
    sim1_mean = 0
    sim2_mean = 0
    random.seed(0)
    for video_id in video_adj_list.keys():
        related_videos = [int(key) for key in video_adj_list[video_id].keys()]
        if len(related_videos) == 0:
            count += 1
            continue
        emb = embeddings[int(video_id)]
        related_emb = embeddings[related_videos]
        sim1 = util.pytorch_cos_sim(emb, related_emb).mean()
        
        related_videos = [random.randint(0, 380000) for _ in video_adj_list[video_id].keys()]
        related_emb = embeddings[related_videos]
        sim2 = util.pytorch_cos_sim(emb, related_emb).mean()
        if count % 10000 == 0 and debug == 1:
            print("Similarity:", sim1)
            print("Similarity random:", sim2)
            print("Similarity gain:", sim1-sim2)
            print("Video id:", id_videos[video_id])
            print("Related video id:", [id_videos[key] for key in video_adj_list[video_id].keys()])
        count += 1
        gain_mean += sim1-sim2
        sim1_mean += sim1
        sim2_mean += sim2

    print("Average gain:", gain_mean/count)
    print("Average sim1:", sim1_mean/count)
    print("Average sim2:", sim2_mean/count)
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

run_related_video_analysis(id_videos, video_adj_list, video_embeddings)
run_related_video_analysis(id_videos, video_adj_list, video_embeddings_des)


VERSION = "_new"
TYPE = ""
embeddings = video_embeddings
with h5py.File(f"../dataset/train_data{VERSION}{TYPE}.hdf5", "r") as train_hf:
        input = np.array(train_hf["input"][:])
        label = np.array(train_hf["label"][:])
        label_type = np.array(train_hf["label_type"][:])
        mask = np.array(train_hf["mask"][:])

M, N = mask.shape

sim_gain_all = 0
sim_gain_all_1 = 0
sim_gain_all_2 = 0
for i in tqdm(range(1000)):
    input_emb = []
    rec_emb = []
    other_emb = []
    input_ids = []
    input_cate = []
    rec_ids = []
    rec_cate = []
    other_ids = []
    for j in range(sum(mask[i])):
        input_emb.append(embeddings[input[i][j]])
        input_ids.append(id_videos[str(input[i][j])])
        input_cate.append(video2category_ids[id_videos[str(input[i][j])]])
        if j == sum(mask[i]) - 1:
            for k in range(len(label_type[i][j])):
                # print(label[i][j][k])
                if label_type[i][j][k] == 1:
                    rec_emb.append(embeddings[label[i][j][k]])
                    rec_ids.append(id_videos[str(label[i][j][k])])
                    rec_cate.append(video2category_ids[id_videos[str(label[i][j][k])]])
                else:
                    other_emb.append(embeddings[random.randint(0, 380000)])
                    other_ids.append(id_videos[str(label[i][j][k])])
            # print(len(rec_emb), len(other_emb))
    sim_gain = []
    sim_gain_1 = []
    sim_gain_2 = []
    # print(input_ids)
    # print(rec_ids)
    # print(input_cate)
    # print(rec_cate)
    j = 0
    for emb in input_emb:
        sim1 = util.pytorch_cos_sim(emb, rec_emb).mean()
        sim2 = util.pytorch_cos_sim(emb, other_emb).mean()
        # print(input_ids[j])
        # print(rec_ids)
        # print(sim1)
        # print(other_ids[:20])
        # print(sim2[:20])
        j += 1
        sim_gain.append(sim1 - sim2)
        sim_gain_1.append(sim1)
        sim_gain_2.append(sim2)
    sim_gain_mean = np.mean(sim_gain)
    sim_gain_all += sim_gain_mean
    sim_gain_all_1 += np.mean(sim_gain_1)
    sim_gain_all_2 += np.mean(sim_gain_2)

print(sim_gain_all / 1000, sim_gain_all_1 / 1000, sim_gain_all_2 / 1000)


