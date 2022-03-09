import os
import json
import subprocess
from tqdm import tqdm
import ray
import logging
import argparse
from constants import *

# VERSION = "rl_reddit_new2"
# VERSION = "final_joint_cate_100_2_test"
VERSION = "final_with_graph"
VERSION = "final_joint_cate_100_2_0.1"
VERSION = "final_joint_cate_103_2_test"
VERSION = "reddit_40"
VERSION = "latest_joint_cate_010"

# VERSION = "reddit_cate_100_2_test"
parser = argparse.ArgumentParser(description='get metadata.')
parser.add_argument('--log', type=str, dest="log_path", help='log path', default=f"./logs/log_metadata_{VERSION}.txt")
parser.add_argument('--data', type=str, dest="sock_puppet_path", help='sock puppet path', default=f"{root_path}/dataset/sock_puppets_{VERSION}.json")
parser.add_argument('--video', type=str, dest="video_id_path", help='video id path', default=f"{root_path}/dataset/video_stat_{VERSION}.json")
parser.add_argument('--metadata', type=str, dest="video_metadata_path", help='video metadata path', default=f"{root_path}/dataset/video_metadata_{VERSION}.json")
args = parser.parse_args()


logging.basicConfig(
    filename=args.log_path,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger() 
logger.setLevel(logging.INFO) 


def get_automatic_captions(js: dict):
    captions = js.get('automatic_captions', {})
    if 'en' not in captions:
        return None
    return captions['en'][0]['url']


def get_subtitles(js: dict):
    subtitles = js.get('subtitles', {})
    if 'en' not in subtitles:
        return None
    return subtitles['en'][0]['url']


@ray.remote
def get_metadata(video_id_list: list):
    ret = {}
    for video_id in tqdm(video_id_list):
        try:
            url = 'https://youtube.com/watch?v=%s' % video_id
            if os.path.exists(f"/scratch/YT_dataset/metadata/{video_id}.json"):
                js = json.load(open(f"/scratch/YT_dataset/metadata/{video_id}.json", "r"))
            else:
                js = json.loads(subprocess.run(['/usr/local/bin/youtube-dl', '-J', url], stdout=subprocess.PIPE).stdout)
                with open(f"/scratch/YT_dataset/metadata/{video_id}.json", "w") as json_file:
                    json.dump(js, json_file)
            metadata = dict(
                title=js.get('title', ''),
                channel_id=js.get('channel_id', ''),
                description=js.get('description', ''),
                view_count=js.get('view_count', ''),
                average_rating=js.get("average_rating", ''),
                thumbnails=','.join([t.get('url') for t in js.get('thumbnails', '')]),
                categories=','.join(js.get('categories', [])),
                subtitles=get_subtitles(js),
                automated_captions=get_automatic_captions(js)
            )
            ret.update({video_id: metadata})

        except:
            ret.update({video_id: {}})

    return ret

def get_all_metadata(
    sock_puppet_path: str="../dataset/sock_puppets_final.json",
    video_id_path: str="../dataset/video_stat_final.json",
    video_metadata_path: str="../dataset/video_metadata_final.json"
):

    # Get video trails
    with open(sock_puppet_path, "r") as json_file:
        data = json.load(json_file)[2]["data"]

    # Parse video trails
    logger.info("Start paring sock puppets.")
    video_ids = {}
    false_cnt = 0
    for i in tqdm(range(len(data))):
        try:
            # Viewed videos
            video_views = data[i]["viewed"]
            for video_id in video_views:
                if video_id not in video_ids.keys():
                    video_ids[video_id] = 0
                video_ids[video_id] += 1

            # History videos
            for video_views in data[i]["homepage"]:
                for video_id in video_views:
                    if video_id not in video_ids.keys():
                        video_ids[video_id] = 0
                    video_ids[video_id] += 1

            # Recommended videos
            rec_trails = data[i]["recommendation_trail"]
            for trail in rec_trails:
                for video_id in trail:
                    if video_id not in video_ids.keys():
                        video_ids[video_id] = 0
                    video_ids[video_id] += 1 
        except:
            false_cnt += 1
            continue
    logger.info("Finish paring sock puppets.")
    logger.info("Miss {} data points.".format(false_cnt))

    # Save all video ids
    with open(video_id_path, "w") as json_file:
        json.dump(video_ids, json_file)
    logger.info("Finish saving ids for videos.")

    # Get video list
    video_id_list = list(video_ids.keys())
    print(len(video_id_list))
    num_cpus = os.cpu_count()
    batch_size = len(video_id_list) // num_cpus + 1

    logger.info("Start getting metadata for videos.")
    ray.init()
    rets = ray.get(
        [get_metadata.remote(video_id_list[i*batch_size: (i+1)*batch_size]) for i in range(num_cpus)]
    )
    ray.shutdown()
    logger.info("Finish getting metadata for videos.")

    # Save all video metadata
    video_metadata = {} #
    # video_metadata = json.load(open(video_metadata_path, "r"))
    for ret in rets:
        video_metadata.update(ret)

    with open(video_metadata_path, "w") as json_file:
        json.dump(video_metadata, json_file)
    logger.info("Finish saving metadata for videos.")

if __name__=="__main__":
    get_all_metadata(
        sock_puppet_path=args.sock_puppet_path,
        video_id_path=args.video_id_path,
        video_metadata_path=args.video_metadata_path
    )