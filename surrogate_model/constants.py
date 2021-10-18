import json

with open("../dataset/home_video_id_sorted_final_filter.json", "r") as json_file:
    data = json.load(json_file)
home_video_id_sorted = [int(key) for key in data.keys()]
all_values = list(data.values())
home_video_value_sorted = [int(key) / sum(data.values()) for key in all_values]

with open("../dataset/video2channel_ids_final.json", "r") as json_file:
    video2channel_ids = json.load(json_file)
    
with open("../dataset/video_ids_final.json", "r") as json_file:
    video_ids = json.load(json_file)
    
video2channel = {}
for video_id in video2channel_ids.keys():
    video2channel[video_ids[video_id]] = video2channel_ids[video_id]
