import pandas as pd
import json

video_des = {}
df = pd.read_csv("../dataset/video-metadata.csv")
for i in range(len(df["video_id"])):
    try:
        video_des[df["video_id"][i]] = df["description"][i].split("\n")[0]
    except:
        video_des[df["video_id"][i]] = df["title"][i]

with open("../dataset/video_description.json", "w") as json_file:
    json.dump(video_des, json_file)
