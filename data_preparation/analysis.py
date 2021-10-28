import json
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='process yt dataset.')
parser.add_argument('--phase', type=int, help='preprocess phase')
args = parser.parse_args()



VERSION = "_reddit"
with open(f"../dataset/sock_puppets{VERSION}.json", "r") as json_file:
    data = json.load(json_file)[2]["data"]
# Parse video trails
rec_video_id = {}
view_video_id = {}
home_video_id = {}
false_cnt = 0
initial_home_video_ids = {}
for i in tqdm(range(len(data))):
    initial_home_video_ids.update(dict(zip(data[i]["initial_homepage"], [1 for _ in range(len(data[i]["initial_homepage"]))])))
print(len(initial_home_video_ids))
for i in tqdm(range(len(data))):
    try:
        # Viewed videos
        video_views = data[i]["viewed"]
        for video_id in video_views:
            if video_id not in view_video_id.keys():
                view_video_id[video_id] = 0
            view_video_id[video_id] += 1
        
        # History videos
        video_views = data[i]["homepage"]
        tmp = {}
        for video_view in video_views:
            for video_id in video_view:
                if video_id in initial_home_video_ids.keys():
                    continue
                if video_id not in home_video_id.keys():
                    home_video_id[video_id] = 0
                if video_id not in tmp.keys():
                    home_video_id[video_id] += 1
                    tmp[video_id] = 1
                else:
                    tmp[video_id] = 1
        # Recommended videos
        rec_trails = data[i]["recommendation_trail"]
        for trail in rec_trails:
            for video_id in trail:
                if video_id not in rec_video_id.keys():
                    rec_video_id[video_id] = 0
                rec_video_id[video_id] += 1 
    except:
        false_cnt += 1
        continue
print("No. viewed videos: {}.".format(len(view_video_id.keys())))
print("No. rec videos: {}.".format(len(rec_video_id.keys())))
print("No. homepage videos: {}.".format(len(home_video_id.keys())))

plt.figure()
viewed_video = np.log10(np.array(sorted(list(view_video_id.values()))))
y = 1. * np.arange(len(viewed_video)) / (len(viewed_video) - 1)
plt.plot(viewed_video, y)
plt.xlabel("No. of occurance time (in log_10)")
plt.ylabel("cdf")
plt.title("No. viewed videos: {}.".format(len(view_video_id.keys())))
plt.savefig(f"./fig/viewed_vidoes{VERSION}.png")

plt.figure()
rec_video = np.log10(np.array(sorted(list(rec_video_id.values()))))
y = 1. * np.arange(len(rec_video)) / (len(rec_video) - 1)
plt.plot(rec_video, y)
plt.xlabel("No. of occurance time (in log_10)")
plt.ylabel("cdf")
plt.title("No. rec videos: {}.".format(len(rec_video_id.keys())))
plt.savefig(f"./fig/rec_vidoes{VERSION}.png")

plt.figure()
home_video = np.log10(np.array(sorted(list(home_video_id.values()))))
y = 1. * np.arange(len(home_video)) / (len(home_video) - 1)
plt.plot(home_video, y)
plt.xlabel("No. of occurance time (in log_10)")
plt.ylabel("cdf")
plt.title("No. homepage videos: {}.".format(len(home_video_id.keys())))
plt.savefig(f"./fig/home_vidoes{VERSION}.png")

# with open(f"../dataset/video_ids{VERSION}.json", "r") as json_file:
#     video_ids = json.load(json_file)

# home_video_id_sorted = {video_ids[k]: v for k, v in sorted(home_video_id.items(), key=lambda item: item[1], reverse=True)}
# with open("../dataset/home_video_id_sorted{VERSION}.json", "w") as json_file:
#     json.dump(home_video_id_sorted, json_file)