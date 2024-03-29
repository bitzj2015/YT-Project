import numpy as np
import ray
import torch
import random
from constants import *
import logging
from worker import *


# Define the environment for training RL agent
class Env(object):
    def __init__(self, env_args, yt_model, denoiser, rl_agent, workers, seed=0, id2video_map=None, use_rand=0, video_by_cate=None):
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
        self.video_by_cate = video_by_cate
        if video_by_cate:
            self.video_cate_set = list(video_by_cate.keys())
        self.use_rand = use_rand
        self.bias_weight = [0 for _ in range(self.env_args.action_dim)]
        self.video_set = [i for i in range(self.env_args.action_dim)]
        if self.use_rand == 2:
            # self.bias_weight = [1 / self.env_args.action_dim for _ in range(self.env_args.action_dim)]
            # self.prev_bias_weight = [i / sum(BIAS_WEIGHT) for i in BIAS_WEIGHT]
            with open(f"./results/bias_weight_{self.env_args.alpha}_{VERSION}.json", "r") as json_file:
                data = json.load(json_file)
            self.bias_weight = list(data)
            # print(len(self.bias_weight))
            sum_w = sum(self.bias_weight)
            self.bias_weight = [i / sum_w for i in self.bias_weight]

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
        
        # print(self.state.size(), self.base_state.size())
        # print(self.cur_cate[0][3:6], self.cur_cate_base[0][3:6])
        # print(self.cur_cate[0][107:110], self.cur_cate_base[0][107:110])

        # Get denoised non-obfuscated recommendations
        # cur_cate_base_pred = self.denoiser.denoiser_model.get_rec(self.base_state, self.state, torch.from_numpy(self.cur_cate).to(self.env_args.device)) # input_vu, input_vo, label_ro

        # Reward for obfuscator
        cur_reward_obfuscator = [ekl_divergence(self.cur_cate[i], self.cur_cate_base[i]) for i in range(len(self.workers))]
        # cur_reward_obfuscator = [((self.cur_cate_base[i] - self.cur_cate[i]) ** 2).sum() for i in range(len(self.workers))]

        # Reward for denoiser
        # cur_reward_denoiser = [-kl_divergence(self.cur_cate_base[i], cur_cate_base_pred[i]) for i in range(len(self.workers))]
        # cur_reward_denoiser = [0 for _ in range(len(self.workers))]
        cur_reward_denoiser = [kl_divergence(self.cur_cate[i], self.cur_cate_base[i]) for i in range(len(self.workers))]

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
            return np.random.choice(self.video_set, len(self.workers), p=self.bias_weight)
        elif self.use_rand == 3:
            video_cates = np.random.choice(self.video_cate_set, len(self.workers))
            return video_cates
            
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
        self.env_args.logger.info(f"start rolling out, length: {self.env_args.rollout_len}")
        for _ in range(self.env_args.rollout_len):
            
            flag = random.random()
            if flag < self.env_args.alpha:
                if self.use_rand != 3:
                    obfuscation_videos = self.get_next_obfuscation_videos()

                    ret = ray.get([self.workers[i].rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                    if not ret[0]:
                        break
                    self.send_reward_to_workers()
                    self.update_rl_agent(reward_only=True)
                    if self.use_rand == 1:
                        rewards = self.get_reward_gain_from_workers()
                        for i in range(len(obfuscation_videos)):
                            self.bias_weight[obfuscation_videos[i]] += max(0, rewards[i])
                else:
                    best_obfuscation_video_cate = []
                    best_reward = []
                    rollout_end = False
                    for t in range(100):
                        obfuscation_video_cate = self.get_next_obfuscation_videos()

                        obfuscation_videos = []
                        for cate in obfuscation_video_cate:
                            obfuscation_videos.append(np.random.choice(self.video_by_cate[cate], 1)[0])
                        if t == 0:
                            ret = ray.get([self.workers[i].rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                            if not ret[0]:
                                rollout_end = True
                                break
                            best_obfuscation_video_cate = obfuscation_video_cate
                        else:
                            ret = ray.get([self.workers[i].re_rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                        self.send_reward_to_workers()
                        rewards = self.get_reward_gain_from_workers()
                        if t == 0:
                            best_reward = rewards
                        else:
                            for i in range(len(obfuscation_video_cate)):
                                if rewards[i] > best_reward[i]:
                                    best_reward[i] = rewards[i]
                                    best_obfuscation_video_cate[i] = obfuscation_video_cate[i]

                    # Greedy search done
                    if rollout_end:
                        break

                    obfuscation_videos = []
                    for cate in best_obfuscation_video_cate:
                        obfuscation_videos.append(np.random.choice(self.video_by_cate[cate], 1)[0])
                    ret = ray.get([self.workers[i].re_rollout.remote(obfuscation_videos[i]) for i in range(len(self.workers))])
                    self.send_reward_to_workers()
                    rewards = self.get_reward_gain_from_workers()
            else:
                
                ret = ray.get([self.workers[i].rollout.remote(-1) for i in range(len(self.workers))])
                if not ret[0]:
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
            ret = self.get_reward_gain_from_workers()
            self.rl_agent.update_rewards(ret)
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
        if save_param and self.use_rand == 0:
            self.rl_agent.save_param()
            self.env_args.logger.info("save rl agent")
        if save_param and self.use_rand == 1:
            with open(f"./results/bias_weight_{self.env_args.alpha}_{VERSION}.json", "w") as json_file:
                json.dump(self.bias_weight, json_file)
