from logging import LogRecord
import os
import json, time
import ray
import logging
import pandas as pd
from tqdm import tqdm
import random


class docker_cfg:
    def __init__(self,**kwargs):
        self.user_desktop = kwargs.get('user_desktop')
        self.shm_size = kwargs.get('shm_size')
        self.user_storage = kwargs.get('user_storage')
        self.image_name = kwargs.get('image')
        self.export = kwargs.get('e', 'DISPLAY=$DISPLAY')
        self.v_docker = f"{self.user_storage}:{self.user_desktop}"
        self.v_x11 = kwargs.get('v_x11', '/tmp/.X11-unix:/tmp/.X11-unix')
        # self.cpuset_cpus = kwargs.get('cpuset', '1')


@ray.remote
class docker_api(object):
    def __init__(self, cmd="", cfg=None, logger=None):
        self.logger = logger
        self.cmd=cmd
        self.cfg=cfg
        self.d_cfg = docker_cfg(
            user_desktop=self.cfg['dump_root'], 
            user_storage=self.cfg['real_root'], 
            image=self.cfg["image"],
            shm_size=self.cfg["shm_size"]
        )

    def run_container(self):
        crawling_cmd = self.cmd

        disable_ipv6 = "--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1 --sysctl net.ipv6.conf.lo.disable_ipv6=1 "
        try:
            # os.system("yes | sudo docker container prune")
            # docker_cmd = f"docker run -v {self.d_cfg.v_docker} -e {self.d_cfg.export} -v {self.d_cfg.v_x11} {disable_ipv6} --memory={self.d_cfg.shm_size} --shm-size={self.d_cfg.shm_size} {self.d_cfg.image_name} {crawling_cmd}"
            docker_cmd = f"docker run -v {self.d_cfg.v_docker} -e {self.d_cfg.export} -v {self.d_cfg.v_x11} {disable_ipv6} {self.d_cfg.image_name} {crawling_cmd}"
            print(f"Run docker container using command {docker_cmd}")
            os.system(docker_cmd)

        except Exception as e:
            print(f"Run container using command {docker_cmd} with exception {e}")
            return 0

        print('Run container successfully.')
        return 1 


def run_docker(user_video_seqs, dump_root="/home/user/Desktop/crawls", real_root="$PWD/docker-volume/crawls_reddit", timeout=38, image="ytdriver:latest", shm_size="512m", logger=None):
    d_api = []
    for user_key in user_video_seqs:
        cfg = {}
        cfg["video_seq"] = user_video_seqs[user_key]
        video_seq = json.dumps(user_video_seqs[user_key])
        cfg["user_id"] = user_key
        cfg["real_root"] = real_root
        cfg["dump_root"] = dump_root
        cfg["image"] = image
        cfg["shm_size"] = shm_size
        cfg["cmd"] = f"timeout {timeout}m /usr/bin/python3 /opt/crawler_eval.py --video-seq '{video_seq}' --save-dir {user_key}"
        d_api.append(docker_api.remote(cmd=cfg["cmd"], cfg=cfg, logger=logger))
        os.system(f"mkdir -p {real_root}/{user_key}")
        os.system(f"touch {real_root}/{user_key}/log.txt")
        os.system(f"chmod -R 777 {real_root}")
    process = [c.run_container.remote() for c in d_api]
    ray.get(process)


if __name__ == "__main__":
    random.seed(0)
    task = "rl_eval"
    batch = []
    if task == "reddit_crawl":
        with open("../dataset/sample_reddit_traces.json", "r") as json_file:
            reddit_user_data = json.load(json_file)
        
        users = list(reddit_user_data.keys())
        sample_users = random.sample(users, 10000)
        count = 0
        
        video_seqs = {}

        for user in sample_users:
            if count < 4650:
                count += 1
                continue
            video_seq = reddit_user_data[user]
            video_seqs[f"{user}_{count}"] = video_seq
            count += 1
            if count % 150 == 0:
                batch.append(video_seqs)
                video_seqs = {}

        logging.basicConfig(
            filename=f"./logs/log.txt",
            filemode='w',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        logger=logging.getLogger() 
        logger.setLevel(logging.INFO)

        for i in tqdm(range(len(batch))):
            ray.init()
            run_docker(batch[i], logger=logger, real_root="$PWD/docker-volume/crawls_reddit", timeout=38)
            ray.shutdown()

    elif task == "rl_eval":
        tag = "final_joint_cate_100_2_test"
        with open(f"../obfuscation/results/test_user_trace_0.2_{tag}_0_new.json", "r") as json_file:
            rl_user_data = json.load(json_file)

        with open(f"../obfuscation/results/test_user_trace_0.2_{tag}_1_new.json", "r") as json_file:
            rand_user_data = json.load(json_file)

        batch = []
        for i in range(38):
            video_seqs = {}
            for j in range(40):
                if i*40+j >= 1500:
                    break
                video_seqs[f"rl_base_{i*40+j}"] = rl_user_data["base"][str(i*40+j)]
                video_seqs[f"rl_obfu_{i*40+j}"] = rl_user_data["obfu"][str(i*40+j)]
                video_seqs[f"rand_base_{i*40+j}"] = rand_user_data["base"][str(i*40+j)]
                video_seqs[f"rand_obfu_{i*40+j}"] = rand_user_data["obfu"][str(i*40+j)]
            batch.append(video_seqs)
    
        logging.basicConfig(
            filename=f"./logs/log.txt",
            filemode='w',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        logger=logging.getLogger() 
        logger.setLevel(logging.INFO)

        for i in tqdm(range(len(batch))):
            ray.init()
            run_docker(batch[i], logger=logger, real_root=f"$PWD/docker-volume/crawls_{tag}", timeout=50)
            ray.shutdown()
