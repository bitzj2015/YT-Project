import json
import numpy as np
import logging
import h5py
from tqdm import tqdm

np.random.seed(0)
logging.basicConfig(
    filename="./logs/log_getdataset.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 

VERSION = "_new"

with open("../dataset/sock-puppets-new.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]

with open(f"../dataset/video_ids{VERSION}.json", "r") as json_file:
    video_ids = json.load(json_file)

def sample_false_label(
    label: list, 
    num_labels: int, 
    max_label_len: int=100
):
    ret = label
    true_label = dict(zip(ret, [0 for _ in range(len(ret))]))
    false_label = list(np.random.randint(num_labels, size=max_label_len+len(label)))

    idx = 0
    while len(ret) < max_label_len:
        if false_label[idx] in true_label.keys():
            idx +=1 
            continue
        ret.append(false_label[idx])
        idx += 1
    return ret


cnt = 0
missed_videos = {}
input_data_all = []
label_data_all = []
label_type_data_all = []
mask_data_all = []

stat_len = {}
max_label_len = 100
max_trail_len = 40
for i in tqdm(range(len(data))):
    try:
        input_data = []
        mask_data = []

        # Get video trails
        video_views = data[i]["viewed"][2:-2].split("\", \"")
        for video_id in video_views:
            # if video_id in video_ids.keys():
            input_data.append(video_ids[video_id])
            mask_data.append(1)

        # Append -1 if the length of trail is smaller than max_trail_len
        if len(input_data) < max_trail_len:
            for _ in range(max_trail_len-len(input_data)):
                input_data.append(-1)
                mask_data.append(0)

        # Get recommended video trails
        rec_trails = data[i]["recommendation_trail"][2:-2].split("], [")
        rec_trails = [trail[1:-1].split("\", \"") for trail in rec_trails]
        label_data = []
        label_type_data = []
        for j in range(len(rec_trails)-1):
            label = []
            label_type = []
            trail = rec_trails[j]

            # Get all the recommended videos each step
            for video_id in trail:
                label.append(video_ids[video_id])
                label_type.append(1)
            
            # Label_type: 0 -> true label, 1 -> false label
            for _ in range(max_label_len-len(label_type)):
                label_type.append(0)

            # Generate false labels
            label = sample_false_label(label, len(video_ids.keys()), max_label_len)
            label_data.append(np.array(label))
            label_type_data.append(np.array(label_type))

        # Get homepage recommendation
        home_rec = data[i]["homepage"][2:-2].split("\", \"")
        home_rec_len = len(home_rec)
        if home_rec_len not in stat_len.keys():
            stat_len[home_rec_len] = 0
        stat_len[home_rec_len] += 1

        # In the last step, we want to predict the homepage videos
        label = []
        label_type = []
        for video_id in home_rec:
            label.append(video_ids[video_id])
            label_type.append(1)

        # Label_type: 0 -> true label, 1 -> false label
        for _ in range(max_label_len-len(label_type)):
            label_type.append(0)

        # Generate false labels
        label = sample_false_label(label, len(video_ids.keys()), max_label_len)
        label_data.append(np.array(label))
        label_type_data.append(np.array(label_type))

        # Append zero matrix if the length of label list is smaller than max_trail_len
        if len(label_data) < max_trail_len:
            for _ in range(max_trail_len-len(label_data)):
                label_data.append(np.array(label)*0)
                label_type_data.append(np.array(label_type)*0)

        input_data_all.append(np.array(input_data))
        label_data_all.append(np.array(label_data))
        label_type_data_all.append(np.array(label_type_data))
        mask_data_all.append(np.array(mask_data))

    except:
        cnt += 1
    
logger.info("Missing {} trails.".format(cnt))
logger.info("Homepage No. recommended videos: {}".format(stat_len))

idx = [i for i in range(len(input_data_all))]
np.random.seed(0)
np.random.shuffle(idx)

train_size = int(len(idx) * 0.8)
input_data_all = np.stack(input_data_all)
label_data_all = np.stack(label_data_all)
label_type_data_all = np.stack(label_type_data_all)
mask_data_all = np.stack(mask_data_all)

logger.info("Input: {}, label: {}, label_type: {}, mask: {}.".format(
    np.shape(input_data_all), 
    np.shape(label_data_all), 
    np.shape(label_type_data_all), 
    np.shape(mask_data_all))
)

hf = h5py.File(f"../dataset/train_data{VERSION}.hdf5", "w")
hf.create_dataset('input', data=input_data_all[idx[:train_size]])
hf.create_dataset('label', data=label_data_all[idx[:train_size]])
hf.create_dataset('label_type', data=label_type_data_all[idx[:train_size]])
hf.create_dataset('mask', data=mask_data_all[idx[:train_size]])
hf.close()

hf = h5py.File(f"../dataset/test_data{VERSION}.hdf5", "w")
hf.create_dataset('input', data=input_data_all[idx[train_size:]])
hf.create_dataset('label', data=label_data_all[idx[train_size:]])
hf.create_dataset('label_type', data=label_type_data_all[idx[train_size:]])
hf.create_dataset('mask', data=mask_data_all[idx[train_size:]])
hf.close()
