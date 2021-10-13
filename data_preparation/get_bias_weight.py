import h5py
import json

VERSION = "_new"
with h5py.File(f"../dataset/train_data{VERSION}.hdf5", "r") as hf:
    label = hf["label"][:]
    label_type = hf["label_type"][:]
    mask = hf["mask"][:]

label_dist = {}
for i in range(len(mask)):
    cur_label_type = label_type[i][sum(mask[i]) - 1]
    cur_label = label[i][sum(mask[i]) - 1]

    for j in range(len(cur_label_type)):
        if cur_label_type[j] == 1:
            if str(cur_label[j]) not in label_dist.keys():
                label_dist[str(cur_label[j])] = 0
            label_dist[str(cur_label[j])] += 1

print(len(label_dist))
label_dist = dict(sorted(label_dist.items(), key=lambda item: item[1], reverse=True))

with open(f"../dataset/home_video_id_sorted{VERSION}.json", "w") as json_file:
    json.dump(label_dist, json_file)
