import sys
sys.path.append('../surrogate_model')
import torch
import torch.optim as optim
from env import *
from agent_graph import *
from config import *
from constants import *
from utils_cate import *
from graph_encoder import GraphEncoder
from graph_aggregator import GraphAggregator
from policy_net_regression_cate import PolicyNetRegression
import argparse
import h5py
import json
import logging
import ray

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default=f"{root_path}/dataset/video_embeddings_{VERSION}_aug.hdf5")
parser.add_argument('--video-id', dest="video_id_path", type=str, default=f"{root_path}/dataset/video_ids_{VERSION}.json")
parser.add_argument('--train-data', dest="train_data_path", type=str, default=f"{root_path}/dataset/train_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--test-data', dest="test_data_path", type=str, default=f"{root_path}/dataset/test_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--agent-path', dest="agent_path", type=str, default="./param/agent.pkl")
parser.add_argument('--alpha', dest="alpha", type=float, default=0.2)
parser.add_argument('--use-rand', dest="use_rand", type=int, default=0)
parser.add_argument('--version', dest="version", type=str, default="reddit")
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('--ytmodel-path', dest="ytmodel_path", type=str, default="/scratch/param/policy_no_graph_edge_reg_new73_100_out_64_001_atten_reddit_filter_aug.pt")
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
video_embeddings = torch.from_numpy(video_embeddings)

# Define environment configuration and rl agent
env_args = EnvConfig(action_dim=video_embeddings.shape[0], device=device, agent_path=args.agent_path, alpha=args.alpha, logger=logger, version=args.version, reward_dim=1)

# Define GNN for yt
yt_agg_video_graph = GraphAggregator(
    video_embeddings=video_embeddings, 
    emb_dim=env_args.emb_dim, 
    add_edge=True,
    device=device
)
yt_video_graph_embeddings = GraphEncoder(
    video_embeddings=video_embeddings, 
    emb_dim=env_args.emb_dim,  
    video_graph_adj_mat=video_graph_adj_mat, 
    aggregator=yt_agg_video_graph,
    device=device
)

yt_model = PolicyNetRegression(
    emb_dim=env_args.emb_dim,
    hidden_dim=128,
    graph_embeddings=yt_video_graph_embeddings,
    video_embeddings=video_embeddings,
    device=device,
    topk=100,
    use_rand=0,
    num_user_state=100
)

yt_model.load_state_dict(torch.load(args.ytmodel_path, map_location=device).state_dict())
yt_model.device = device
yt_model.video_embeddings = video_embeddings.to(device)
yt_model.graph_embeddings.device = device
yt_model.graph_embeddings.aggregator.device = device

# # Load yt surrogate model
# yt_model = torch.load(
#     args.ytmodel_path, 
#     map_location=device
# ).to(device)
# yt_model.device = device
# yt_model.video_embeddings = video_embeddings.to(device)
# yt_model.graph_embeddings.device = device
# yt_model.graph_embeddings.aggregator.device = device

train_loader, test_loader, val_loader = load_dataset(
    train_data_path=args.train_data_path,
    test_data_path=args.test_data_path,
    batch_size=64,
    logger=logger
)
# State tracker
stat = {
    "train_last_acc": 0, "train_last_acc_ch": 0, "train_last_count": 0, "train_loss_pos": 0, "train_loss_neg": 0, 
    "test_last_acc": 0, "test_last_acc_ch": 0, "test_last_count": 0, "test_loss_pos": 0, "test_loss_neg": 0
}
logger.info("load model")
# Testing
stat = run_regression_epoch(model=yt_model, dataloader=test_loader, mode="test", optimizer=None, ep=0, stat=stat, logger=logger, use_graph=True)


# Define GNN 
agg_video_graph = GraphAggregator(
    video_embeddings=video_embeddings, 
    emb_dim=env_args.emb_dim, 
    add_edge=True,
    device=device
)
video_graph_embeddings = GraphEncoder(
    video_embeddings=video_embeddings, 
    emb_dim=env_args.emb_dim,  
    video_graph_adj_mat=video_graph_adj_mat, 
    aggregator=agg_video_graph,
    device=device
)

# Define rl model
rl_model = A2Clstm(env_args, video_embeddings.to(env_args.device), video_graph_embeddings, with_graph=True).to(device)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)

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
    workers = [RolloutWorker.remote(env_args, i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, rl_agent, workers, seed=0, id2video_map=ID2VIDEO, use_graph=True)
    
    # Start RL agent training loop
    losses = []
    rewards = []
    test_losses = []
    test_rewards = []
    # Start  training
    best_reward = -100
    for ep in range(50):
        env.rl_agent.train()
        for i in range(train_inputs.shape[0] // env_args.num_browsers):
            ray.get([env.workers[j].update_user_videos.remote(train_inputs[i * env_args.num_browsers + j].tolist()) for j in range(env_args.num_browsers)])
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
            ray.get([env.workers[j].update_user_videos.remote(test_inputs[i * env_args.num_browsers + j].tolist()) for j in range(env_args.num_browsers)])
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
    workers = [RolloutWorker.remote(env_args, i) for i in range(env_args.num_browsers)]
    env = Env(env_args, yt_model, rl_agent, workers, seed=0, id2video_map=ID2VIDEO, use_rand=args.use_rand)

    # Start testing
    env.rl_agent.eval()
    test_results = {"base": {}, "obfu": {}}
    user_count = 0
    random.seed(0)
    np.random.seed(0)
    for ep in range(1):
        for i in range(test_inputs.shape[0] // env_args.num_browsers):
            ray.get([env.workers[j].update_user_videos.remote(test_inputs[i * env_args.num_browsers + j].tolist()) for j in range(env_args.num_browsers)])
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

    

