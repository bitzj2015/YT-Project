# from transformers import BertModel, BertConfig
# model = BertModel.from_pretrained('./models/bert-base-uncased')

import json
import numpy as np
import h5py
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import ray

emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')

VERSION = "_final"
TYPE = ""
MAX_LEN = 384

with open(f"../dataset/video_metadata{VERSION}.json", "r") as json_file:
    video_metadata = json.load(json_file)

with open(f"../dataset/video_stat{VERSION}.json", "r") as json_file:
    video_stat = json.load(json_file)

id_videos = dict(zip([i for i in range(len(video_stat.keys()))], video_stat.keys()))
with open(f"../dataset/id_videos{VERSION}.json", "w") as json_file:
    json.dump(id_videos, json_file)

video_ids = dict(zip(video_stat.keys(), [i for i in range(len(video_stat.keys()))]))
with open(f"../dataset/video_ids{VERSION}.json", "w") as json_file:
    json.dump(video_ids, json_file)

embeddings = []
video_texts = []

cnt = 0
cnt2 = 0
for idx in tqdm(sorted(id_videos.keys())):
    video_id = id_videos[idx]
    try:
        video_texts.append(json.load(open(f"/project/kpsounis_171/dataset/trans_parsed/{video_id}.json", "r"))[video_id])
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
hf = h5py.File(f"../dataset/video_embeddings{VERSION}{TYPE}.hdf5", "w")
hf.create_dataset('embeddings', data=embeddings)
hf.create_dataset('video_ids', data=list(video_ids.keys()))
hf.close()