import numpy as np
import ray
import logging

# Define ray worker, each of which simulates one user.
@ray.remote
class RolloutWorker(object):
    def __init__(self, env_args, user_id):
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
        self.user_videos = []
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
        
    def update_user_videos(self, user_videos):
        self.user_videos = user_videos
        
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
            self.logger.info("User 0, cur_reward: {}".format(str(self.cur_rewards)))
    
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