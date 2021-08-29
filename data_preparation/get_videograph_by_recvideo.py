import json
import argparse
import logging
from tqdm import tqdm
import numpy as np
import os
import ray

parser = argparse.ArgumentParser(description='build vide graph.')
parser.add_argument('--phase', type=int, help='preprocess phase')
args = parser.parse_args()

logging.basicConfig(
    filename="./logs/log_videograph.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


@ray.remote
def parse_recvideo(recvideo_file_list: list, video_ids: dict):
    video_video_edge = {}
    stat = {"avg_num_edges": [], "avg_hit_rate": []}
    for recvideo_file in recvideo_file_list:
        try:
            cnt_all = 1e-10
            cnt = 0
            with open(f"./recvideos/{recvideo_file}", "r") as text_file:
                video_id = recvideo_file.split(".")[0]
                video_video_edge[video_id] = {}
                for line in text_file.readlines():
                    cnt_all += 1
                    recvideo = json.loads(line)
                    recvideo_id = recvideo["videoId"]
                    if recvideo_id not in video_ids.keys():
                        # the recvideo is not in our video candidates
                        continue
                    else:
                        video_video_edge[video_id][recvideo_id] = 1
                        cnt += 1
            stat["avg_num_edges"].append(len(video_video_edge[video_id].keys()))
            stat["avg_hit_rate"].append(cnt/cnt_all)
        except:
            continue
    print("avg_num_edges: {}, avg_hit_rate: {}.".format(np.mean(stat["avg_num_edges"]), np.mean(stat["avg_hit_rate"])))
    return video_video_edge


VERSION = "_new"

with open(f"../dataset/video_ids{VERSION}.json", "r") as json_file:
    video_ids = json.load(json_file)

num_cpu = os.cpu_count()

logger.info("Start building graph edges.")
recvideo_file_list = os.listdir("./recvideos")
batch_size = len(recvideo_file_list) // num_cpu + 1

ray.init()
results = ray.get(
    [parse_recvideo.remote(recvideo_file_list[i*batch_size: (i+1)*batch_size], video_ids) for i in range(num_cpu)]
)
ray.shutdown()
video_video_edge = {}
for res in results:
    video_video_edge.update(res)

with open(f"../dataset/video_video_edge{VERSION}.json", "w") as json_file:
    json.dump(video_video_edge, json_file)

video_adj_list = dict(zip([i for i in range(len(video_ids.keys()))], [{} for _ in range(len(video_ids.keys()))]))
for video_id in tqdm(video_video_edge.keys()):
    for recvideo_id in video_video_edge[video_id].keys():
        video_adj_list[video_ids[video_id]][video_ids[recvideo_id]] = 1

with open(f"../dataset/video_adj_list{VERSION}.json", "w") as json_file:
    json.dump(video_adj_list, json_file)

print(len(video_video_edge.keys()), len(video_adj_list.keys()))
