import json
USE_RAND=2

with open("../dataset/home_video_id_sorted.json", "r") as json_file:
    home_video_id_sorted = json.load(json_file)
home_video_id_sorted = [int(key) for key in home_video_id_sorted.keys()]