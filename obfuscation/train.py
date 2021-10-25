import sys
sys.path.append('../surrogate_model')
import torch
import torch.optim as optim
from env import *
from agent import *
from config import *
import argparse
import h5py
import json

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default="../dataset/video_embeddings_final_aug.hdf5")
parser.add_argument('--video-id', dest="video_id_path", type=str, default="../dataset/video_ids_final_filter.json")
parser.add_argument('--train-data', dest="train_data_path", type=str, default="../dataset/train_data_final_filter.hdf5")
parser.add_argument('--test-data', dest="test_data_path", type=str, default="../dataset/test_data_final.hdf5")
parser.add_argument('--agent-path', dest="agent_path", type=str, default="./param/agent.pkl")
parser.add_argument('--alpha', dest="alpha", type=float, default=0.2)
parser.add_argument('--version', dest="version", type=str, default="all")
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('--ytmodel-path', dest="ytmodel_path", type=str, default="/scratch/param/policy_no_graph_edge_reg_new73_100_out_64_001_atten_final_filter_aug.pt")
args = parser.parse_args()

# Check whether cuda is available
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)

# Load video embeddings
with h5py.File(args.video_emb_path, "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")

# Load yt surrogate model
yt_model = torch.load(
    args.ytmodel_path, 
    map_location=device
).to(device)

# Define environment configuration and rl agent
env_args = EnvConfig(action_dim=video_embeddings.shape[0], device=device, agent_path=args.agent_path, alpha=args.alpha)
video_embeddings = torch.from_numpy(video_embeddings).to(env_args.device)
yt_model.device = device
yt_model.video_embeddings = video_embeddings.to(device)
yt_model.graph_embeddings.device = device
yt_model.graph_embeddings.aggregator.device = device
rl_model = A2Clstm(env_args, video_embeddings).to(device)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)

# Start ray  
ray.init()



if not args.eval:
    # Load training inputs
    with h5py.File(args.train_data_path, "r") as train_hf:
        train_inputs = np.array(train_hf["input"][:])
    print(train_inputs.shape, video_embeddings.shape[0])
    
    # Load testing data
    with h5py.File(args.test_data_path, "r") as test_hf:
        test_inputs = np.array(test_hf["input"][:])
    print(test_inputs.shape, video_embeddings.shape[0])
    
    # Initialize envrionment and workers
    workers = [RolloutWorker.remote(env_args, train_inputs[i].tolist(), i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, rl_agent, workers, seed=0)
    
    # Start RL agent training loop
    losses = []
    rewards = []
    test_losses = []
    test_rewards = []
    # Start  training
    for ep in range(50):
        env.rl_agent.train()
        for i in range(train_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = train_inputs[i * env_args.num_browsers + j].tolist()
            try:
                # One episode training
                env.start_env()
                loss, reward = env.rollout()
                losses.append(loss)
                rewards.append(reward)
                
                if i % 10 == 0:
                    print(f"Train epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
                env.stop_env()
            except:
                continue
        with open(f"./results/train_log_{args.alpha}_{args.version}.json", "w") as json_file:
            json.dump({"loss": losses, "reward": rewards}, json_file)
            
        # Start testing
        env.rl_agent.eval()
        for i in range(test_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = test_inputs[i * env_args.num_browsers + j].tolist()
            try:
                # One episode training
                env.start_env()
                loss, reward = env.rollout(train_rl=False)
                test_losses.append(loss)
                test_rewards.append(reward)
                
                if i % 10 == 0:
                    print(f"Test epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
                env.stop_env(save_param=False)
            except:
                continue
        with open(f"./results/eval_log_{args.alpha}_{args.version}.json", "w") as json_file:
            json.dump({"loss": test_losses, "reward": test_rewards}, json_file)
            
else:
    losses = []
    rewards = []
    # Load testing data
    with h5py.File(args.test_data_path, "r") as test_hf:
        test_inputs = np.array(test_hf["input"][:])
    print(test_inputs.shape, video_embeddings.shape[0])
        
    # Load pretrained rl agent
    print("loading model parameters")
    rl_agent.model.load_state_dict(torch.load(args.agent_path))
    
    # Initialize envrionment and workers
    workers = [RolloutWorker.remote(env_args, test_inputs[i].tolist(), i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, rl_agent, workers, seed=0)

    # Start testing
    env.rl_agent.eval()
    for ep in range(1):
        for i in range(test_inputs.shape[0] // env_args.num_browsers):
            for j in range(env_args.num_browsers):
                env.workers[j].user_videos = test_inputs[i * env_args.num_browsers + j].tolist()
            try:
                # One episode training
                env.start_env()
                loss, reward = env.rollout(train_rl=False)
                losses.append(loss)
                rewards.append(reward)
                
                if i % 10 == 0:
                    print(f"Test epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
                env.stop_env(save_param=False)
            except:
                continue
        with open(f"./results/test_log_{args.alpha}_{args.version}.json", "w") as json_file:
            json.dump({"loss": losses, "reward": rewards}, json_file)
ray.shutdown()

    

