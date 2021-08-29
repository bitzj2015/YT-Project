import json
import argparse
import logging
from tqdm import tqdm
import numpy as np
import h5py
import os
import ray
from sqlitedict import SqliteDict

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
def parse_comment(comment_file_list: list, worker_id: str):
    video_author_edge = {}
    author_video_edge = {}
    for comment_file in tqdm(comment_file_list):
        try:
            with open(f"./comments/{comment_file}") as text_file:
                video_id = comment_file.split(".")[0]
                for line in text_file.readlines():
                    comment = json.loads(line)
                    author_id = comment["channel"] # author's channel id
                    # author_id = comment["author"] # author's name
                    if video_id not in video_author_edge.keys():
                        video_author_edge[video_id] = {}
                    video_author_edge[video_id][author_id] = 1
                    if author_id not in author_video_edge.keys():
                        author_video_edge[author_id] = {}
                    author_video_edge[author_id][video_id] = 1
        except:
            continue
    print(len(video_author_edge.keys()), len(author_video_edge.keys()))

    with open(f"../dataset/graph/video_author_edge{worker_id}.json", "w") as json_file:
        json.dump(video_author_edge, json_file)
    with open(f"../dataset/graph/author_video_edge{worker_id}.json", "w") as json_file:
        json.dump(author_video_edge, json_file)


TAG = "_new"
num_task = 4
num_cpu = os.cpu_count() // 4

if args.phase == 0:
    logger.info("Start building graph edges.")
    comment_file_list = os.listdir("./comments")
    
    task_size = len(comment_file_list) // num_task + 1
    for count in range(num_task):
        logger.info(f"Start task {count}.")
        comment_file_list_cur = comment_file_list[count*task_size: (count+1)*task_size]
        batch_size = len(comment_file_list_cur) // num_cpu + 1

        ray.init()
        ray.get(
            [parse_comment.remote(comment_file_list_cur[i*batch_size: (i+1)*batch_size], f"{TAG}_{count}_{i}") for i in range(num_cpu)]
        )
        ray.shutdown()


elif args.phase == 1:
    file_id = []
    for i in range(num_task):
        for j in range(num_cpu):
            file_id.append(f"{TAG}_{i}_{j}")

    author_video_edge_all = {}
    for file_i in tqdm(file_id):
        with open(f"/scratch/graph/author_video_edge{file_i}.json", "r") as json_file:
            author_video_edge = json.load(json_file)

        for author_id in author_video_edge.keys():
            if author_id not in author_video_edge_all.keys():
                author_video_edge_all[author_id] = {}
            author_video_edge_all[author_id].update(author_video_edge[author_id])

    author_ids = list(author_video_edge_all.keys())
    for author_id in tqdm(author_ids):
        if len(author_video_edge_all[author_id].keys()) == 1:
            del author_video_edge_all[author_id]

    with open(f"../dataset/graph/author_video_edge{TAG}_all.json", "w") as json_file:
        json.dump(author_video_edge_all, json_file)


elif args.phase == 2:
    with open(f"../dataset/graph/author_video_edge{TAG}_all.json", "r") as json_file:
        author_video_edge_all = json.load(json_file)

    num_split = 200
    batch_size = len(author_video_edge_all.keys()) // num_split + 1
    
    count = 0
    author_video_edge = {}
    num_task = 0
    for author_id in tqdm(author_video_edge_all.keys()):
        if count == batch_size:
            with open(f"/scratch/graph/author_video_edge{TAG}_{num_task}.json", "w") as json_file:
                json.dump(author_video_edge, json_file)
            count = 0
            num_task += 1
            author_video_edge = {}

        author_video_edge[author_id] = author_video_edge_all[author_id]
        count += 1

    with open(f"/scratch/graph/author_video_edge{TAG}_{num_task}.json", "w") as json_file:
        json.dump(author_video_edge, json_file)


elif args.phase == 3:
    
    mydict = SqliteDict('/scratch/author2video.sqlite', autocommit=True)
    with open(f"../dataset/graph/author_video_edge{TAG}_all.json", "r") as json_file:
        author_video_edge = json.load(json_file)
    for author_id in tqdm(author_video_edge.keys()):
        mydict[author_id] = author_video_edge[author_id]
    mydict.close()

    del author_video_edge
    mydict = SqliteDict('/scratch/video2author.sqlite', autocommit=True)
    with open(f"/scratch/graph/video_author_edge{TAG}_all.json", "r") as json_file:
        video_author_edge = json.load(json_file)
    for video_id in tqdm(video_author_edge.keys()):
        mydict[video_id] = video_author_edge[video_id]
    mydict.close()
    

elif args.phase == 4:
    logger.info("Start loading video_author_edge and author_video_edge.")
    video_author_edge = SqliteDict('/scratch/video2author.sqlite')
    with open(f"../dataset/graph/author_video_edge{TAG}_all.json", "r") as json_file:
        author_video_edge = json.load(json_file)
    video_video_edge = SqliteDict('/scratch/video2video.sqlite', autocommit=True)

    logger.info("Complete loading author_video_edge.")
    with open(f"../dataset/video_ids{TAG}.json", "r") as json_file:
        video_ids = json.load(json_file)
    logger.info("Complete loading video_author_edge and author_video_edge.")
    logger.info("{} videos in total.".format(len(video_ids.keys())))
    
    num_videos = len(list(video_ids.keys()))
    count = 0
    for video_id_row, video_edge in tqdm(video_author_edge.iteritems()):
        for author_id in video_edge.keys():
            row_id = video_ids[video_id_row]
            try: 
                if len(author_video_edge[author_id]) == 1:
                    continue
            except:
                continue
            video_video_edge[row_id].update(author_video_edge[author_id])

    logger.info("Missing {} videos, in total {} videos.".format(count, num_videos))
    video_author_edge.close()
    video_video_edge.close()

elif args.phase == 5:
    with h5py.File("../dataset/video_adj_mat.hdf5", "r") as hf:
        adj_mat = np.array(hf["adj_mat"])
        print(np.shape(adj_mat))
        print(np.sum(adj_mat))
        print(np.sum(adj_mat > 0))
        print(np.sum(adj_mat > 10))
        print(np.sum(adj_mat > 100))
        print(np.sum(adj_mat > 1000))
        print(np.sum(adj_mat > 10000))
        print(np.max(adj_mat))