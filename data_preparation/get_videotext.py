import json
import ray
import os
import subprocess
import logging
from tqdm import tqdm
import random


TAG = "_final"

logging.basicConfig(
    filename="./logs/log_videotext.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


@ray.remote
def get_video_transcript(video_text_url: list):
    for video_id in tqdm(video_text_url):
        subprocess.run(["youtube-dl", "--write-auto-sub", "--skip-download", f"https://www.youtube.com/watch?v={video_id}", "--output", f"../dataset/trans_en/{video_id}"], stdout=subprocess.PIPE).stdout

def get_all_video_transcript(
    video_metadata_path: str=f"../dataset/video_metadata{TAG}.json"
):
    with open(video_metadata_path, "r") as json_file:
        video_metadata = json.load(json_file)

    video_ids = dict(zip(list(video_metadata.keys()), [0 for _ in range(len(video_metadata.keys()))]))
    print("all videos: {}".format(len(video_ids.keys())))
    for filename in os.listdir("../dataset/trans_en"):
        video_id = filename.split(".")[0]
        if video_id in video_ids.keys():
            del video_ids[video_id]

    print("remaining videos: {}".format(len(video_ids.keys())))
    video_id_list = list(video_ids.keys())
    num_cpus = os.cpu_count()
    batch_size = len(video_id_list) // num_cpus + 1

    logger.info("Start getting text.")
    # ray.init()
    # ray.get(
    #     [get_video_transcript.remote(video_id_list[i*batch_size: (i+1)*batch_size]) for i in range(num_cpus)]
    # )
    # ray.shutdown()

if __name__=="__main__":
    get_all_video_transcript(f"../dataset/video_metadata{TAG}.json")