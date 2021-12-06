import numpy as np
import ray
import torch
import random
import json
from constants import *
import logging

# Define ray worker, each of which simulates one user.
@ray.remote
class RolloutWorker(object):
    def __init__(self, env_args, user_videos, user_id):
        self.env_args = env_args
        self.pre_rewards = [0 for _ in range(self.env_args.reward_dim)]
        self.cur_rewards = [0 for _ in range(self.env_args.reward_dim)]
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
        self.user_id = user_id
        logging.basicConfig(
            filename=f"./logs/log_train_regression_{self.env_args.alpha}_{self.env_args.version}.txt",
            filemode='a',
            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
            datefmt='%H:%M:%S',
            level=logging.INFO
        )
        self.logger=logging.getLogger() 
        self.logger.setLevel(logging.INFO) 

    def initial_profile(self, initial_len=5):
        self.watch_history_base = self.user_videos[: initial_len]
        self.watch_history = self.user_videos[: initial_len]
        self.watch_history_type = [0 for _ in range(initial_len)]
        self.user_step += initial_len
        self.step += initial_len

    def update_reward(self, cur_reward):
        self.pre_rewards = self.cur_rewards
        self.cur_rewards = cur_reward
        if self.user_id == 0:
            self.logger.info("cur_reward: {}".format(str(self.cur_rewards)))
    
    def rollout(self, obfuscation_video=-1):
        if self.user_step >= len(self.user_videos):
            return False
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
        return True

    def get_state(self, his_len=10):
        return np.array(self.watch_history[:])
    
    def get_base_state(self, his_len=10):
        return np.array(self.watch_history_base[:])
        
    def get_watch_history(self):
        return (self.watch_history_base, self.watch_history)

    def get_reward(self):
        return self.cur_rewards
    
    def get_reward_gain(self):
        return self.cur_rewards - self.pre_rewards
    
    def clear_worker(self):
        self.pre_rewards = 0
        self.cur_rewards = 0
        self.watch_history = []
        self.watch_history_base = []
        self.watch_history_type = []
        self.cur_rec_base = {}
        self.cur_rec = {}
        self.global_rec_base = {}
        self.global_rec = {}
        self.user_step = 0
        self.step = 0
        if self.user_id == 0:
            self.logger.info("clear worker")


# Define the environment for training RL agent
class Env(object):
    def __init__(self, env_args, yt_model, denoiser, rl_agent, workers, seed=0, id2video_map=None, use_rand=0):
        self.env_args = env_args
        self.all_rewards = []
        self.all_reward_gains = []
        self.all_watch_history = []
        self.all_watch_history_base = []
        self.cur_cate = []
        self.cur_cate_base = []
        self.yt_model = yt_model
        self.denoiser = denoiser
        self.rl_agent = rl_agent
        self.workers = workers
        self.seed = seed
        self.id2video_map = id2video_map
        self.use_rand = use_rand
        self.bias_weight = []
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
        all_watch_history = ray.get([worker.get_watch_history.remote() for worker in self.workers])
        for (watch_history_base, watch_history) in all_watch_history:
            self.all_watch_history_base.append(watch_history_base)
            self.all_watch_history.append(watch_history)
        
    def get_reward_from_workers(self):
        self.all_rewards = ray.get([worker.get_reward.remote() for worker in self.workers])
        return np.array(self.all_rewards).reshape(-1)
    
    def get_reward_gain_from_workers(self):
        self.all_reward_gains = ray.get([worker.get_reward_gain.remote() for worker in self.workers])
        return np.array(self.all_reward_gains).reshape(-1)

    def send_reward_to_workers(self):
        # Get obfuscated personas
        self.state = self.get_state_from_workers()
        self.state = torch.from_numpy(self.state).to(self.env_args.device)

        # Get non-obfuscated personas
        self.base_state = self.get_base_state_from_workers()
        self.base_state = torch.from_numpy(self.base_state).to(self.env_args.device)

        # Get obfuscated recommendations
        self.cur_cate = self.yt_model.get_rec(self.state, topk=100)

        # Get non-obfuscated recommendations
        self.cur_cate_base = self.yt_model.get_rec(self.base_state, topk=100)

        # Get denoised non-obfuscated recommendations
        cur_cate_base_pred = self.denoiser.denoiser_model.get_rec(self.base_state, self.state, torch.from_numpy(self.cur_cate).to(self.env_args.device)) # input_vu, input_vo, label_ro

        # Reward for obfuscator
        cur_reward_obfuscator = [kl_divergence(self.cur_cate_base[i], self.cur_cate[i]) for i in range(len(self.workers))]

        # Reward for denoiser
        cur_reward_denoiser = [-kl_divergence(self.cur_cate_base[i], cur_cate_base_pred[i]) for i in range(len(self.workers))]

        # Total rewards
        self.env_args.logger.info("KL distance of obfuscation: {}, denoiser: {}".format(np.mean(cur_reward_obfuscator), np.mean(cur_reward_denoiser)))
        cur_reward = [self.env_args.reward_w[0] * cur_reward_obfuscator[i] + self.env_args.reward_w[1] * cur_reward_denoiser[i] for i in range(len(self.workers))]
        
        # Send reward back to worker
        ray.get([self.workers[i].update_reward.remote(cur_reward[i]) for i in range(len(self.workers))])
        self.cur_cate = list(self.cur_cate)
        self.cur_cate_base = list(self.cur_cate_base)

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

    def update_denoiser(self, dataloader, train_denoiser=True):
        if train_denoiser:
            loss, kl_div = self.denoiser.train(dataloader)
            self.env_args.logger.info(f"Train denoiser, loss: {loss}, kl_div: {kl_div}")
        else:
            loss, kl_div = self.denoiser.eval(dataloader)
            self.env_args.logger.info(f"Test denoiser, loss: {loss}, kl_div: {kl_div}")
        
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
        self.cur_cate_base = []
        self.cur_cate = []
        self.env_args.logger.info("clear env")
        if save_param:
            self.rl_agent.save_param()
            self.env_args.logger.info("save rl agent")
        # if self.use_rand == 2:
        #     with open(f"./results/bias_weight_new.json", "w") as json_file:
        #         json.dump(self.bias_weight, json_file)
