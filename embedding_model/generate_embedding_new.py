# from transformers import BertModel, BertConfig
# model = BertModel.from_pretrained('./models/bert-base-uncased')

import json
import numpy as np
import h5py
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from constants import root_path

emb_model = SentenceTransformer('all-MiniLM-L6-v2')

VERSION = "final_joint_cate_100_2_test"
VERSION = "final_with_graph"
VERSION = "final_joint_cate_100_2_0.1"
VERSION = "final_joint_cate_103_2_test"
# VERSION = "reddit_cate_100_2_test"
VERSION = "40"
# VERSION = "latest_joint_cate_010"

TYPE = "_large"
MAX_LEN = 256

with open(f"{root_path}/dataset/video_metadata_{VERSION}_large.json", "r") as json_file:
    video_metadata = json.load(json_file)

with open(f"{root_path}/dataset/video_stat_{VERSION}_large.json", "r") as json_file:
    video_stat = json.load(json_file)

id_videos = dict(zip([i for i in range(len(video_stat.keys()))], video_stat.keys()))
with open(f"{root_path}/dataset/id_videos_{VERSION}.json", "w") as json_file:
    json.dump(id_videos, json_file)

video_ids = dict(zip(video_stat.keys(), [i for i in range(len(video_stat.keys()))]))
with open(f"{root_path}/dataset/video_ids_{VERSION}.json", "w") as json_file:
    json.dump(video_ids, json_file)

embeddings = []
video_texts = []

cnt = 0
cnt2 = 0
for idx in tqdm(sorted(id_videos.keys())):
    video_id = id_videos[idx]
    try:
        video_texts.append(json.load(open(f"{root_path}/dataset/trans_parsed/{video_id}.json", "r"))[video_id])
        cnt2 += 1
    except:
        if "description" in video_metadata[video_id].keys():
            video_texts.append(video_metadata[video_id]["description"].split("\n")[0])

        elif "title" in video_metadata[video_id].keys():
            video_texts.append(video_metadata[video_id]["title"])
            
        else:
            video_texts.append(" ")
            cnt += 1
print("Missing {} videos' metadata, {} with transcripts.".format(cnt, cnt2))

def get_embedding(video_text_list, emb_model):
    embeddings = []
    for i in tqdm(range(len(video_text_list))):
        tokens = video_text_list[i].split(" ")
        text_chunks = [" ".join(tokens[j*MAX_LEN: (j+1)*MAX_LEN]) for j in range(len(tokens) // MAX_LEN + 1)]
        embeddings.append(np.mean(emb_model.encode(text_chunks), axis=0).reshape(1,-1))
    return embeddings

num_cpu = os.cpu_count() 
batch_size = len(video_texts) // num_cpu + 1
embeddings = get_embedding(video_texts, emb_model)

embeddings = np.concatenate(embeddings, axis=0)
print(np.shape(embeddings))
hf = h5py.File(f"{root_path}/dataset/video_embeddings_{VERSION}{TYPE}.hdf5", "w")
hf.create_dataset('embeddings', data=embeddings)
hf.create_dataset('video_ids', data=list(video_ids.keys()))
hf.close()
