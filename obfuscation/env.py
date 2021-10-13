import numpy as np
import torch
from utils import get_features
import os
import json
from copy import deepcopy
import ray
import random

@ray.remote
class RolloutWorker(object):
    def __init__(self, env_args, user_videos):
        self.env_args = env_args
        self.pre_rewards = {"removed": 0, "added": 0}
        self.cur_rewards = {"removed": 0, "added": 0}
        self.watch_history = []
        self.watch_history_base = []
        self.watch_history_type = []
        self.cur_rec_base = {}
        self.cur_rec = {}
        self.global_rec_base = {}
        self.global_rec = {}
        self.user_step = 0
        self.step = 0
        self.user_videos = user_videos

    def initial_profile(self, initial_len=5):
        self.watch_history_base = self.user_videos[: initial_len]
        self.watch_history = self.user_videos[: initial_len]
        self.watch_history_type = [0 for _ in range(initial_len)]
        self.user_step += initial_len
        self.step += initial_len

    def update_reward(self, cur_rec_base, cur_rec):
        self.cur_rec_base = dict(zip(cur_rec_base, [0 for _ in range(len(cur_rec_base))]))
        self.cur_rec = dict(zip(cur_rec, [0 for _ in range(len(cur_rec))]))
        self.global_rec_base.update(self.cur_rec_base)
        self.global_rec.update(self.cur_rec)
        self.pre_rewards = self.cur_rewards
        self.cur_reward = {"removed": 0, "added": 0}
        for video in self.global_rec_base.keys():
            if video not in self.global_rec.keys():
                self.cur_reward["removed"] += 1
        for video in self.global_rec.keys():
            if video not in self.global_rec_base.keys():
                self.cur_reward["added"] += 1
    
    def rollout(self, obfuscation_video=-1):
        if obfuscation_video == -1:
            video = self.user_videos[self.user_step]
            self.watch_history_base.append(video)
            self.watch_history_type.append(0)
            self.user_step += 1
        else:
            video = obfuscation_video
            self.watch_history_type.append(1)
        self.watch_history.append(video)
        self.step += 1

    def get_state(self, his_len=10):
        return np.array(self.watch_history[:his_len])
    
    def get_base_state(self, his_len=10):
        return np.array(self.watch_history_base[:his_len])

    def get_reward(self):
        return self.cur_reward["removed"] + self.cur_reward["added"]
    
    def get_reward_gain(self):
        return self.cur_reward["removed"] + self.cur_reward["added"] - self.pre_reward["removed"] - self.pre_reward["removed"]

    def clear_worker(self):
        self.pre_rewards = {"removed": 0, "added": 0}
        self.cur_rewards = {"removed": 0, "added": 0}
        self.watch_history = []
        self.watch_history_base = []
        self.watch_history_type = []
        self.cur_rec_base = {}
        self.cur_rec = {}
        self.global_rec_base = {}
        self.global_rec = {}
        self.user_step = 0
        self.step = 0


class Env(object):
    def __init__(self, env_args, yt_model, rl_agent, workers, seed=0):
        self.env_args = env_args
        self.pre_rewards = [0 for _ in range(self.env_args.num_browsers)]
        self.cur_rewards = [0 for _ in range(self.env_args.num_browsers)]
        self.watch_history = None
        self.watch_history_base = None
        self.watch_history_type = None
        self.cur_rec_base = [{} for _ in range(self.env_args.num_browsers)]
        self.cur_rec = [{} for _ in range(self.env_args.num_browsers)]
        self.global_rec_base = [{} for _ in range(self.env_args.num_browsers)]
        self.global_rec = [{} for _ in range(self.env_args.num_browsers)]
        self.yt_model = yt_model
        self.rl_agent = rl_agent
        self.workers = workers
        self.seed = seed

    def start_env(self):
        ray.init()
        ray.get([worker.initial_profile.remote(self.env_args.his_len) for worker in self.workers])

    def stop_env(self):
        self.clear_env()
        ray.shutdown()
        
    def get_reward_from_workers(self):
        self.all_rewards = ray.get([worker.get_reward.remote(self.env_args.his_len) for worker in self.workers])
        return np.array(self.all_rewards)
    
    def get_reward_gain_from_workers(self):
        self.all_reward_gains = ray.get([worker.get_reward_gain.remote(self.env_args.his_len) for worker in self.workers])
        return np.array(self.all_reward_gains)

    def send_reward_to_workers(self):
        self.state = self.get_state_from_workers()
        self.base_state = self.get_base_state_from_workers()
        cur_rec = self.yt_model.get_rec(self.state)
        cur_rec_base = self.yt_model.get_rec(self.base_state)
        ray.get([worker.update_reward(cur_rec_base, cur_rec) for worker in self.workers])

    def get_next_obfuscation_videos(self):
        self.state = self.get_state_from_workers()
        return self.rl_agent.take_action(self.state)

    def get_state_from_workers(self):
        self.state = np.stack(ray.get([worker.get_state.remote() for worker in self.workers]))
        return self.state

    def get_base_state_from_workers(self):
        self.base_state = np.stack(ray.get([worker.get_base_state.remote() for worker in self.workers]))
        return self.base_state

    def rollout(self):
        for _ in range(self.env_args.rollout_len):
            if random.random() < self.env_args.alpha:
                obfuscation_videos = self.get_next_obfuscation_videos()
                ray.get([self.workers[i].rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                self.send_reward_to_workers()
                self.update_rl_agent(reward_only=True)
            else:
                ray.get([self.workers[i].rollout.remote(-1) for i in range(len(self.workers))])
        self.update_rl_agent(reward_only=False)

    def update_rl_agent(self, reward_only=True):
        if reward_only:
            self.rl_agent.update_rewards(self.get_reward_gain_from_workers())
        else:
            self.rl_agent.update_model()

    def clear_env(self):
        self.all_rewards = []
        self.all_reward_gains = []
        self.all_watch_history_base = []
        self.all_watch_history = []
        self.rl_agent.save_param()