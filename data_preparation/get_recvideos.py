from tqdm import tqdm
import json
import subprocess
import ray
import os
import argparse

parser = argparse.ArgumentParser(description='get recvideo.')
parser.add_argument('--start', type=int, dest="start", help='start point', default=0)
parser.add_argument('--end', type=int, dest="end", help='end_point', default=1000000)
parser.add_argument('--version', type=str, dest="version", help='dataset version', default="reddit")
args = parser.parse_args()

VERSION = args.version
@ray.remote
def get_recvideos(video_id_list: list):
    for video_id in tqdm(video_id_list):
        try:
            # youtube-comment-downloader
            subprocess.run(["./downloader_recvideos.py", "--youtubeid", video_id, "--output", f"./recvideos/{video_id}.json"], stdout=subprocess.PIPE).stdout
        except:
            continue

def get_recvideos_all(
    video_id_path=f"../dataset/video_ids_{VERSION}.json"
):
    with open(video_id_path, "r") as json_file:
        video_ids = json.load(json_file)
    
    video_id_list = list(video_ids.keys())
    count = 0
    count_remain = 0
    video_ids_new = {}
    for video_id in video_id_list:
        if os.path.isfile(f"./recvideos/{video_id}.json"):
            count += 1
            video_ids_new[video_id] = 0
        else:
            video_ids_new[video_id] = 0
            count_remain += 1
    print("No. of existing videos: {}, No. of remaining videos: {}".format(count, count_remain))
    with open(f"../dataset/video_ids_{VERSION}_remain.json", "w") as json_file:
        json.dump(video_ids_new, json_file)
    
    video_id_list = list(video_ids_new.keys())[args.start:]
    num_cpus = os.cpu_count()
    batch_size = len(video_id_list) // num_cpus + 1

    ray.init()
    ray.get(
        [get_recvideos.remote(video_id_list[i*batch_size: (i+1)*batch_size]) for i in range(num_cpus)]
    )
    ray.shutdown()

get_recvideos_all(f"../dataset/video_ids_{VERSION}.json")