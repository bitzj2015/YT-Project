import json
import numpy as np
import h5py
from constants import root_path

VERSION = "_final_joint_cate_100_2_0.1"
VERSION = "_final_joint_cate_103_2_test"
VERSION = "_40_June"
VERSION = "_reddit_40_new"
VERSION = "_realuser_all"
LOAD_METADATA = True

with open(f"{root_path}/dataset/video_metadata{VERSION}_new.json", "r") as json_file:
    video_metadata = json.load(json_file)

with open(f"{root_path}/dataset/video_stat{VERSION}.json", "r") as json_file:
    video_stat = json.load(json_file)

print(len(video_metadata))
id_videos = dict(zip([i for i in range(len(video_stat.keys()))], video_stat.keys()))
with open(f"{root_path}/dataset/id_videos{VERSION}.json", "w") as json_file:
    json.dump(id_videos, json_file)

video_ids = dict(zip(video_stat.keys(), [i for i in range(len(video_stat.keys()))]))
with open(f"{root_path}/dataset/video_ids{VERSION}.json", "w") as json_file:
    json.dump(video_ids, json_file)

with open(f"{root_path}/dataset/topic/tag2class{VERSION}2.json", "r") as json_file:
    tag2class = json.load(json_file)

with open(f"{root_path}/dataset/topic/class2id2.json", "r") as json_file:
    class2id = json.load(json_file)

categories = []
view_counts = []
average_ratings = []
if LOAD_METADATA:
    with open(f"{root_path}/dataset/metadata_40.json", "r") as json_file:
        metadata = json.load(json_file)
        category_ids = metadata["category"]
        mean_average_ratings = metadata["mean_average_ratings"]
        std_average_ratings = metadata["std_average_ratings"]
        mean_view_counts = metadata["mean_view_counts"]
        std_view_counts = metadata["std_view_counts"]
else:
    category_ids = {"none": 0, "*": 1}
video2channel = {}
channel2video = {}
for idx in sorted(id_videos.keys()):
    video_id = id_videos[idx]
    category = [0 for _ in range(len(class2id))]
    try:
        tags = video_metadata[video_id]["tags"].split(",")
        for tag in tags:
            if tag == "":
                continue

            c = class2id[tag2class[tag][0]]
            conf = tag2class[tag][1]
            if conf > 0:
                category[c] = 1

    except:
        category = [0 for _ in range(len(class2id))]
    categories.append(category)
    
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

print(category_ids)
with open(f"{root_path}/dataset/category_ids_latest.json", "w") as json_file:
    json.dump(category_ids, json_file)
    
categories = np.array(categories)
average_ratings = np.array(average_ratings)
view_counts = np.array(view_counts)

# max_cate = np.max(categories) + 2
# I = np.eye(max_cate)
# one_hot_categories = I[categories]

if LOAD_METADATA:
    average_ratings = (average_ratings - mean_average_ratings) / (std_average_ratings + 1e-10)
    view_counts = (view_counts - mean_view_counts) / (std_view_counts + 1e-10)
    print(mean_average_ratings, std_average_ratings, mean_view_counts, std_view_counts)
    print(np.mean(average_ratings), np.std(average_ratings), np.max(average_ratings), np.min(average_ratings))
    print(np.mean(view_counts), np.std(view_counts), np.max(view_counts), np.min(view_counts))
else:
    with open(f"{root_path}/dataset/metadata{VERSION}.json", "w") as json_file:
        json.dump({
            "category": category_ids,
            "mean_average_ratings": np.mean(average_ratings),
            "std_average_ratings": np.std(average_ratings),
            "mean_view_counts": np.mean(view_counts),
            "std_view_counts": np.std(view_counts)},
            json_file)
    average_ratings = (average_ratings - np.mean(average_ratings)) / (np.std(average_ratings) + 1e-10)
    view_counts = (view_counts - np.mean(view_counts)) / (np.std(view_counts) + 1e-10)
    print(np.mean(average_ratings), np.std(average_ratings), np.max(average_ratings), np.min(average_ratings))
    print(np.mean(view_counts), np.std(view_counts), np.max(view_counts), np.min(view_counts))


with h5py.File(f"{root_path}/dataset/video_embeddings{VERSION}.hdf5", "r") as hf:
    embeddings = hf["embeddings"][:]
    videoIds = hf["video_ids"][:]
print(embeddings.shape)
print(isinstance(embeddings[0][0], np.float32))
augmented_embeddings = np.concatenate([embeddings, categories, average_ratings.reshape(-1,1), view_counts.reshape(-1, 1)], axis=1)
print(augmented_embeddings.shape)

hf = h5py.File(f"{root_path}/dataset/video_embeddings{VERSION}_aug2.hdf5", "w")
hf.create_dataset('embeddings', data=augmented_embeddings.astype("float32"))
hf.create_dataset('video_ids', data=videoIds)
hf.close()
print(isinstance(augmented_embeddings[0][0], np.float32))

with open(f"{root_path}/dataset/video2channel{VERSION}.json", "w") as json_file:
    json.dump(video2channel, json_file)

with open(f"{root_path}/dataset/channel2video{VERSION}.json", "w") as json_file:
    json.dump(channel2video, json_file)

channel_ids = dict(zip(list(channel2video.keys()), [i for i in range(len(channel2video))]))
video2channel_ids = {}
for video_id in video2channel.keys():
    video2channel_ids[video_id] = channel_ids[video2channel[video_id]]

with open(f"{root_path}/dataset/channel_ids{VERSION}.json", "w") as json_file:
    json.dump(channel_ids, json_file)

with open(f"{root_path}/dataset/video2channel_ids{VERSION}.json", "w") as json_file:
    json.dump(video2channel_ids, json_file)



