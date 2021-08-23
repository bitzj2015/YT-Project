import json
import numpy as np
import h5py
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

VERSION = "_new"

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
descriptions = []

cnt = 0
for idx in sorted(id_videos.keys()):
    video_id = id_videos[idx]
    if "description" in video_metadata[video_id].keys():
        descriptions.append(video_metadata[video_id]["description"].split("\n")[0])

    elif "title" in video_metadata[video_id].keys():
        descriptions.append(video_metadata[video_id]["title"])
        
    else:
        descriptions.append(" ")
        cnt += 1
print("Missing {} videos' metadata.".format(cnt))

batch_size = 1000
for i in tqdm(range(0,len(descriptions),batch_size)):
    embeddings.append(emb_model.encode(descriptions[i:i+batch_size]))

embeddings = np.concatenate(embeddings, axis=0)
hf = h5py.File(f"../dataset/video_embeddings{VERSION}.hdf5", "w")
hf.create_dataset('embeddings', data=embeddings)
hf.create_dataset('video_ids', data=list(video_ids.keys()))
hf.close()