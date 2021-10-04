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

categories = []
view_counts = []
average_ratings = []
category_ids = {"none": 0, "*": 1}
video2channel = {}
channel2video = {}
for idx in sorted(id_videos.keys()):
    video_id = id_videos[idx]
    try:
        cate = video_metadata[video_id]["categories"]
        if cate != "":
            if cate not in category_ids.keys():
                category_ids[cate] = category_ids["*"]
                category_ids["*"] += 1
            categories.append(category_ids[cate])
        else:
            categories.append(0)
    except:
        categories.append(0)
    
    try:
        view_counts.append(int(video_metadata[video_id]["view_count"]))
    except:
        view_counts.append(0)
    
    try:
        average_ratings.append(int(video_metadata[video_id]["average_rating"]))
    except:
        average_ratings.append(0)

    try: 
        channel_id = video_metadata[video_id]["channel_id"]
        video2channel[video_id] = channel_id
        if channel_id not in channel2video.keys():
            channel2video[channel_id] = {}
        channel2video[channel_id][video_id] = 1
    except:
        video2channel[video_id] = ""


categories = np.array(categories)
average_ratings = np.array(average_ratings)
view_counts = np.array(view_counts)

max_cate = np.max(categories) + 1
I = np.eye(max_cate)
one_hot_categories = I[categories]

average_ratings = (average_ratings - np.mean(average_ratings)) / (np.std(average_ratings) + 1e-10)
view_counts = (view_counts - np.mean(view_counts)) / (np.std(view_counts) + 1e-10)
print(np.mean(average_ratings), np.std(average_ratings), np.max(average_ratings), np.min(average_ratings))
print(np.mean(view_counts), np.std(view_counts), np.max(view_counts), np.min(view_counts))

with h5py.File(f"../dataset/video_embeddings{VERSION}.hdf5", "r") as hf:
    embeddings = hf["embeddings"][:]
    videoIds = hf["video_ids"][:]
print(embeddings.shape)
print(isinstance(embeddings[0][0], np.float32))
augmented_embeddings = np.concatenate([embeddings, one_hot_categories, average_ratings.reshape(-1,1), view_counts.reshape(-1, 1)], axis=1)
print(augmented_embeddings.shape)

hf = h5py.File(f"../dataset/video_embeddings{VERSION}_aug.hdf5", "w")
hf.create_dataset('embeddings', data=augmented_embeddings.astype("float32"))
hf.create_dataset('video_ids', data=videoIds)
hf.close()
print(isinstance(augmented_embeddings[0][0], np.float32))

with open(f"../dataset/video_video_edge{VERSION}_w.json", "r") as json_file:
    video_video_edge = json.load(json_file)

video_adj_list = dict(zip([i for i in range(len(video_ids.keys()))], [{} for _ in range(len(video_ids.keys()))]))
for video_id in tqdm(video_video_edge.keys()):
    for recvideo_id in video_video_edge[video_id].keys():
        video_adj_list[video_ids[video_id]][video_ids[recvideo_id]] = video_video_edge[video_id][recvideo_id]
    channel_id = video2channel[video_id]
    if channel_id != "":
        for same_ch_video in channel2video[channel_id].keys():
            video_adj_list[video_ids[video_id]][video_ids[same_ch_video]] = 1 
print(len(channel2video.keys()))
with open(f"../dataset/video_adj_list{VERSION}_w_aug.json", "w") as json_file:
    json.dump(video_adj_list, json_file)

with open(f"../dataset/video2channel{VERSION}.json", "w") as json_file:
    json.dump(video2channel, json_file)

with open(f"../dataset/channel2video{VERSION}.json", "w") as json_file:
    json.dump(channel2video, json_file)

