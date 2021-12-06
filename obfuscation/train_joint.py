import sys
sys.path.append('../surrogate_model')
sys.path.append('../denoiser')
import torch
import torch.optim as optim
from env_joint import *
from agent_joint import *
from config import *
from constants import *
from denoiser import *
import argparse
import h5py
import json
import logging

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default=f"../dataset/video_embeddings_{VERSION}_aug.hdf5")
parser.add_argument('--video-id', dest="video_id_path", type=str, default=f"../dataset/video_ids_{VERSION}.json")
parser.add_argument('--train-data', dest="train_data_path", type=str, default=f"../dataset/train_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--test-data', dest="test_data_path", type=str, default=f"../dataset/test_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--agent-path', dest="agent_path", type=str, default="./param/agent.pkl")
parser.add_argument('--denoiser-path', dest="denoiser_path", type=str, default="./param/denoiser.pkl")
parser.add_argument('--alpha', dest="alpha", type=float, default=0.2)
parser.add_argument('--use-rand', dest="use_rand", type=int, default=0)
parser.add_argument('--version', dest="version", type=str, default="reddit")
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('--ytmodel-path', dest="ytmodel_path", type=str, default="/scratch/param/policy_no_graph_edge_reg_new73_100_out_64_001_atten_final_filter_cate2.pt")
args = parser.parse_args()

logging.basicConfig(
    filename=f"./logs/log_train_regression_{args.alpha}_{args.version}.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


# Check whether cuda is available
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
logger.info(f"Use cuda: {use_cuda}")

# Load video embeddings
with h5py.File(args.video_emb_path, "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")

# Load yt surrogate model
yt_model = torch.load(
    args.ytmodel_path, 
    map_location=device
).to(device)

# Define environment configuration and rl agent
env_args = EnvConfig(action_dim=video_embeddings.shape[0], device=device, agent_path=args.agent_path, alpha=args.alpha, logger=logger, version=args.version)
video_embeddings = torch.from_numpy(video_embeddings).to(env_args.device)
yt_model.device = device
yt_model.video_embeddings = video_embeddings.to(device)
yt_model.graph_embeddings.device = device
yt_model.graph_embeddings.aggregator.device = device
rl_model = A2Clstm(env_args, video_embeddings).to(device)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)

# Define denoiser
denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device)
denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=env_args.denoiser_lr)
denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)

# Start ray  
ray.init()

if not args.eval:
    # Load training inputs
    with h5py.File(args.train_data_path, "r") as train_hf:
        train_inputs = np.array(train_hf["input"][:])
    env_args.logger.info("Training data size: {}, No. of videos: {}.".format(train_inputs.shape, video_embeddings.shape[0]))
    
    # Load testing data
    with h5py.File(args.test_data_path, "r") as test_hf:
        test_inputs = np.array(test_hf["input"][:])
    env_args.logger.info("Testing data size: {}, No. of videos: {}.".format(test_inputs.shape, video_embeddings.shape[0]))

    # Initialize envrionment and workers
    workers = [RolloutWorker.remote(env_args, train_inputs[i].tolist(), i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, denoiser, rl_agent, workers, seed=0, id2video_map=ID2VIDEO)
    
    # Start RL agent training loop
    losses = []
    rewards = []
    test_losses = []
    test_rewards = []
    # Start  training
    
    best_reward = -100
    for ep in range(50):
        # Update denoiser
        base_persona, obfu_persona, base_rec, obfu_rec = [], [], [] ,[]
        max_len = 0
        for i in range(train_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = train_inputs[i * env_args.num_browsers + j].tolist()
            env.start_env()
            _, _ = env.rollout(train_rl=False)
            env.get_watch_history_from_workers()
            base_persona += env.all_watch_history_base
            obfu_persona += env.all_watch_history
            base_rec += env.cur_cate_base
            obfu_rec += env.cur_cate
            # print(len(base_persona))
            env.stop_env(save_param=False)
        for item in obfu_persona:
            max_len = max(max_len, len(item))
        denoiser_train_loader, denoiser_test_loader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, env_args.num_browsers, max_len)
        for round in range(5):
            env_args.logger.info(f"Training denoiser round: {round}")
            env.update_denoiser(denoiser_train_loader)
        env_args.logger.info(f"Testing denoiser")
        env.update_denoiser(denoiser_test_loader, train_denoiser=False)

        # Update obfuscator (rl agent)
        env.rl_agent.train()
        for i in range(train_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = train_inputs[i * env_args.num_browsers + j].tolist()
            # try:
            # One episode training
            env.start_env()
            loss, reward = env.rollout()
            losses.append(loss)
            rewards.append(reward)
            
            if i % 10 == 0:
                env_args.logger.info(f"Train epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
            if best_reward < np.mean(rewards[-112:]):
                env.stop_env()
                best_reward = np.mean(rewards[-112:])
            else:
                env.stop_env(save_param=False)
            # except:
            #     continue
        with open(f"./results/train_log_{args.alpha}_{args.version}.json", "w") as json_file:
            json.dump({"loss": losses, "reward": rewards}, json_file)
            
        # Start testing
        env.rl_agent.eval()
        for i in range(test_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = test_inputs[i * env_args.num_browsers + j].tolist()
            # try:
            # One episode training
            env.start_env()
            loss, reward = env.rollout(train_rl=False)
            test_losses.append(loss)
            test_rewards.append(reward)
            
            if i % 10 == 0:
                env_args.logger.info(f"Test epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
            env.stop_env(save_param=False)
            # except:
            #     continue
        with open(f"./results/eval_log_{args.alpha}_{args.version}.json", "w") as json_file:
            json.dump({"loss": test_losses, "reward": test_rewards}, json_file)
            
else:
    losses = []
    rewards = []
    # Load testing data
    with h5py.File(args.test_data_path, "r") as test_hf:
        test_inputs = np.array(test_hf["input"][:])
    env_args.logger.info("Testing data size: {}, No. of videos: {}.".format(test_inputs.shape, video_embeddings.shape[0]))
        
    # Load pretrained rl agent
    env_args.logger.info("loading model parameters")
    rl_agent.model.load_state_dict(torch.load(args.agent_path, map_location=device))
    
    # Initialize envrionment and workers
    workers = [RolloutWorker.remote(env_args, test_inputs[i].tolist(), i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, rl_agent, workers, seed=0, id2video_map=ID2VIDEO, use_rand=args.use_rand)

    # Start testing
    env.rl_agent.eval()
    test_results = {"base": {}, "obfu": {}}
    user_count = 0
    random.seed(0)
    np.random.seed(0)
    for ep in range(1):
        for i in range(test_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = test_inputs[i * env_args.num_browsers + j].tolist()
            # try:
            # One episode training
            env.start_env()
            loss, reward = env.rollout(train_rl=False)
            losses.append(loss)
            rewards.append(reward)
            
            if i % 10 == 0:
                env_args.logger.info(f"Test epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
            rets = env.get_watch_history_from_workers()
            for ret in rets:
                test_results["base"][str(user_count)] = ret["base"]
                test_results["obfu"][str(user_count)] = ret["obfu"]
                user_count += 1
            env.stop_env(save_param=False)
            # except:
            #     continue
        with open(f"./results/test_log_{args.alpha}_{args.version}_{args.use_rand}.json", "w") as json_file:
            json.dump({"loss": losses, "reward": rewards}, json_file)
        with open(f"./results/test_user_trace_{args.alpha}_{args.version}_{args.use_rand}_new.json", "w") as json_file:
            json.dump(test_results, json_file)
            
ray.shutdown()

    

