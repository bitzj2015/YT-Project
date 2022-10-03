import numpy as np
import ray
import torch
import random
from constants import *
import logging
from worker import *


# Define the environment for training RL agent
class Env(object):
    def __init__(self, env_args, yt_model, rl_agent, workers, seed=0, id2video_map=None, use_rand=0, use_graph=False):
        self.env_args = env_args
        self.all_rewards = []
        self.all_reward_gains = []
        self.watch_history = []
        self.watch_history_base = []
        self.yt_model = yt_model
        self.rl_agent = rl_agent
        self.workers = workers
        self.seed = seed
        self.id2video_map = id2video_map
        self.use_rand = use_rand
        self.bias_weight = []
        self.use_graph = use_graph
        self.video_set = [i for i in range(self.env_args.action_dim)]
        if self.use_rand == 2:
            self.bias_weight = [1 / self.env_args.action_dim for _ in range(self.env_args.action_dim)]
            self.prev_bias_weight = [i / sum(BIAS_WEIGHT) for i in BIAS_WEIGHT]

    def start_env(self):
        # random.seed(self.seed)
        ray.get([worker.initial_profile.remote(self.env_args.his_len) for worker in self.workers])

    def stop_env(self, save_param=True):
        self.clear_env(save_param=save_param)
        ray.get([worker.clear_worker.remote() for worker in self.workers])
    
    def get_watch_history_from_workers(self):
        rets = []
        all_watch_history = ray.get([worker.get_watch_history.remote() for worker in self.workers])
        for (watch_history_base, watch_history) in all_watch_history:
            ret = {}
            ret["base"] = [self.id2video_map[str(index)] for index in watch_history_base]
            ret["obfu"] = [self.id2video_map[str(index)] for index in watch_history]
            rets.append(ret) 
        return rets
        
    def get_reward_from_workers(self):
        self.all_rewards = ray.get([worker.get_reward.remote() for worker in self.workers])
        return np.array(self.all_rewards).reshape(-1)
    
    def get_reward_gain_from_workers(self):
        self.all_reward_gains = ray.get([worker.get_reward_gain.remote() for worker in self.workers])
        return np.array(self.all_reward_gains).reshape(-1)

    def send_reward_to_workers(self):
        self.state = self.get_state_from_workers()
        self.base_state = self.get_base_state_from_workers()
        cur_cate = self.yt_model.get_rec(torch.from_numpy(self.state).to(self.env_args.device), topk=100, with_graph=self.use_graph)
        cur_cate_base = self.yt_model.get_rec(torch.from_numpy(self.base_state).to(self.env_args.device), topk=100, with_graph=self.use_graph)
        cur_reward = [kl_divergence(cur_cate_base[i], cur_cate[i]) for i in range(len(self.workers))]
        self.env_args.logger.info("KL distance between embeddings: {}, {}, {}".format(np.mean(cur_reward), len(cur_reward), cur_cate.shape))
        ray.get([self.workers[i].update_reward.remote(cur_reward[i]) for i in range(len(self.workers))])

    def get_next_obfuscation_videos(self, terminate=False):
        self.state = self.get_state_from_workers()
        if self.use_rand == 0:
            return self.rl_agent.take_action(torch.from_numpy(self.state).to(self.env_args.device), terminate=terminate)
        elif self.use_rand == 1:
            return np.random.choice(self.video_set, len(self.workers))
        elif self.use_rand == 2:
            # normalized_bias_weight = [item / sum(self.bias_weight) for item in self.bias_weight]
            return np.random.choice(self.video_set, len(self.workers), p=self.prev_bias_weight)
            
    def get_state_from_workers(self):
        self.state = np.stack(ray.get([worker.get_state.remote(self.env_args.his_len) for worker in self.workers]))
        return self.state

    def get_base_state_from_workers(self):
        self.base_state = np.stack(ray.get([worker.get_base_state.remote(self.env_args.his_len) for worker in self.workers]))
        return self.base_state

    def rollout(self, train_rl=True):
        self.env_args.logger.info("start rolling out")
        for _ in range(self.env_args.rollout_len):
            flag = random.random()
            if flag < self.env_args.alpha:
                obfuscation_videos = self.get_next_obfuscation_videos()
                ret = ray.get([self.workers[i].rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                if not ret:
                    break
                self.send_reward_to_workers()
                self.update_rl_agent(reward_only=True)
                if self.use_rand == 2:
                    rewards = self.get_reward_gain_from_workers()
                    for i in range(len(obfuscation_videos)):
                        self.bias_weight[obfuscation_videos[i]] += max(0, rewards[i])
            else:
                ret = ray.get([self.workers[i].rollout.remote(-1) for i in range(len(self.workers))])
                if not ret:
                    break
        self.get_next_obfuscation_videos(terminate=True)
        self.env_args.logger.info("end rolling out")
        if train_rl and self.use_rand == 0:
            loss = self.update_rl_agent(reward_only=False)
        else:
            loss = 0
            self.rl_agent.clear_actions()
        mean_rewards = np.mean(self.get_reward_from_workers())
        return loss, mean_rewards

    def update_rl_agent(self, reward_only=True):
        if reward_only:
            self.rl_agent.update_rewards(self.get_reward_gain_from_workers())
        else:
            loss, _ = self.rl_agent.update_model()
            self.rl_agent.clear_actions()
            return loss

    def clear_env(self, save_param=True):
        self.all_rewards = []
        self.all_reward_gains = []
        self.all_watch_history_base = []
        self.all_watch_history = []
        self.env_args.logger.info("clear env")
        if save_param:
            self.rl_agent.save_param()
            self.env_args.logger.info("save rl agent")
        # if self.use_rand == 2:
        #     with open(f"./results/bias_weight_new.json", "w") as json_file:
        #         json.dump(self.bias_weight, json_file)
