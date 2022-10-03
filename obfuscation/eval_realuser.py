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
import ray

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default=f"{root_path}/dataset/video_embeddings_{VERSION}_aug.hdf5")
parser.add_argument('--video-id', dest="video_id_path", type=str, default=f"{root_path}/dataset/video_ids_{VERSION}.json")
parser.add_argument('--test-data', dest="test_data_path", type=str, default=f"{root_path}/dataset/sock_puppets_{VERSION}{TAG}.json")
parser.add_argument('--agent-path', dest="agent_path", type=str, default="./param/agent.pkl")
parser.add_argument('--denoiser-path', dest="denoiser_path", type=str, default="./param/denoiser_0.2_v2_kldiv.pkl")
parser.add_argument('--alpha', dest="alpha", type=float, default=0.2)
parser.add_argument('--use-rand', dest="use_rand", type=int, default=0)
parser.add_argument('--version', dest="version", type=str, default="realuser")
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('--ytmodel-path', dest="ytmodel_path", type=str, default="../surrogate_model/param/policy_v2_kldiv.pt")
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

with h5py.File(args.video_emb_path.replace("realuser_all", "40_June"), "r") as hf_emb:
    video_embeddings_aug = hf_emb["embeddings"][:].astype("float32")

video_embeddings = np.concatenate([video_embeddings, video_embeddings_aug], axis=0)

with open(f"{root_path}/dataset/video_ids_40_June.json", "r") as json_file:
    VIDEO_IDS_AUG = json.load(json_file)

AUG_LEN = len(VIDEO_IDS.keys())
print(AUG_LEN, len(VIDEO_IDS_AUG))
print(VIDEO_IDS["zYLVpPgGrec"])
ID2VIDEO = {}
for key in VIDEO_IDS_AUG.keys():
    if key in VIDEO_IDS.keys():
        ID2VIDEO[str(VIDEO_IDS_AUG[key] + AUG_LEN)] = key
        continue
    else:
        VIDEO_IDS[key] = VIDEO_IDS_AUG[key] + AUG_LEN


all_ids = list(VIDEO_IDS.keys())
for i in range(len(VIDEO_IDS.keys())):
    ID2VIDEO[str(VIDEO_IDS[all_ids[i]])] = all_ids[i]


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
yt_model.graph_embeddings.video_embeddings = video_embeddings.to(device)
yt_model.eval()
rl_model = A2Clstm(env_args, video_embeddings).to(device)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)

with open(f"./results/bias_weight_{env_args.alpha}_40_June.json", "r") as json_file:
    data = json.load(json_file)

data = [0 for _ in range(AUG_LEN)] + data
print(len(data), video_embeddings.shape[0])

with open(f"./results/bias_weight_{env_args.alpha}_{VERSION}.json", "w") as json_file:
    json.dump(data, json_file)


# Define denoiser
denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device)
denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=env_args.denoiser_lr)
denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)

# Start ray  
ray.init()

losses = []
rewards = []

# Load testing data
with open(args.test_data_path, "r") as test_file:
    test_data = json.load(test_file)[2]["data"]
env_args.logger.info("Testing data size: {}, No. of videos: {}.".format(len(test_data), video_embeddings.shape[0]))

test_inputs = []
for i in range(len(test_data)):
    test_inputs.append([VIDEO_IDS[video] for video in test_data[i]["viewed"]])
    assert len(test_inputs[-1]) == 40

# Load pretrained rl agent
env_args.logger.info("loading model parameters")
rl_agent.model.load_state_dict(torch.load(args.agent_path, map_location=device))
rl_agent.model.video_embeddings = video_embeddings.to(device)


# Initialize envrionment and workers
workers = [RolloutWorker.remote(env_args, i) for i in range(env_args.num_browsers)]
env = Env(env_args, yt_model, denoiser, rl_agent, workers, seed=0, id2video_map=ID2VIDEO, use_rand=args.use_rand)
# env.denoiser.denoiser_model.load_state_dict(torch.load(args.denoiser_path, map_location=device))

# Start testing
env.rl_agent.eval()
test_results = {"base": {}, "obfu": {}}
user_count = 0
random.seed(0)
np.random.seed(0)
for ep in range(1):
    # Update denoiser
        
    for i in range(len(test_inputs) // env_args.num_browsers):
        ray.get([env.workers[j].update_user_videos.remote(test_inputs[i * env_args.num_browsers + j]) for j in range(env_args.num_browsers)])
        # try:
        # One episode training
        env.start_env()
        loss, reward = env.rollout(train_rl=False)
        losses.append(loss)
        rewards.append(reward)
        
        if i % 10 == 0:
            env_args.logger.info(f"Test epoch: {ep}, episode: {i}, loss: {loss}, reward: {reward}")
        env.get_watch_history_from_workers()
        for j in range(len(env.all_watch_history_base)):
            test_results["base"][str(user_count)] = [ID2VIDEO[str(video_id)] for video_id in env.all_watch_history_base[j]]
            test_results["obfu"][str(user_count)] = [ID2VIDEO[str(video_id)] for video_id in env.all_watch_history[j]]
            user_count += 1
        all_watch_history = ray.get([worker.get_watch_history.remote() for worker in env.workers])
        print(all_watch_history[0])
        env.stop_env(save_param=False)
        # except:
        #     continue
    with open(f"./results/test_log_{args.alpha}_{args.version}_{args.use_rand}.json", "w") as json_file:
        json.dump({"loss": losses, "reward": rewards}, json_file)
    with open(f"./results/test_user_trace_{args.alpha}_{args.version}_{args.use_rand}_new.json", "w") as json_file:
        json.dump(test_results, json_file)
            
ray.shutdown()

    

