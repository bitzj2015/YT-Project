import numpy as np
import json

with open("../dataset/video_ids_new.json", "r") as json_file:
    video_ids = json.load(json_file)

video_id_list = list(video_ids.keys())[:1000]
recvideo_dict = dict(zip(video_id_list, [{} for _ in range(1000)]))

miss = 0
for video_id in video_id_list:
    for i in range(9):
        try:
            if i < 8:
                filename = f"./recvideo_{i}/{video_id}.json"
            else:
                filename = f"./recvideos/{video_id}.json"
            with open(filename, "r") as file:
                line_num = 0
                for line in file.readlines():
                    line_num += 1
                    rec = json.loads(line)["videoId"]
                    if rec not in recvideo_dict[video_id].keys():
                        recvideo_dict[video_id][rec] = {"ranks": [], "mean": [], "var": []}
                    recvideo_dict[video_id][rec]["ranks"].append(line_num)
        except:
            miss += 1
    for rec in recvideo_dict[video_id].keys():
        recvideo_dict[video_id][rec]["mean"] = np.mean(recvideo_dict[video_id][rec]["ranks"])
        recvideo_dict[video_id][rec]["var"] = np.std(recvideo_dict[video_id][rec]["ranks"])

with open("recvideo_dict.json", "w") as file:
    json.dump(recvideo_dict, file)

