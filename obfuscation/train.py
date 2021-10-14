import sys
sys.path.append('../surrogate_model')
import torch
import torch.optim as optim
from env import *
from agent import *
from config import *
import argparse
import h5py

parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default="../dataset/video_embeddings_final_aug.hdf5")
parser.add_argument('--video-id', dest="video_id_path", type=str, default="../dataset/video_ids_final.json")
parser.add_argument('--train-data', dest="train_data_path", type=str, default="../dataset/train_data_final.hdf5")
parser.add_argument('--test-data', dest="test_data_path", type=str, default="../dataset/test_data_final.hdf5")
args = parser.parse_args()

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

with h5py.File(args.video_emb_path, "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")

yt_model = torch.load(
    "/scratch/param/policy_no_graph_edge_reg_new73_100_out_64_001_atten_final_aug.pt", 
    map_location=device
)


env_args = EnvConfig(action_dim=video_embeddings.shape[0], device=device)
video_embeddings = torch.from_numpy(video_embeddings).to(env_args.device)
yt_model.device = device
yt_model.video_embeddings = video_embeddings.to(device)
yt_model.graph_embeddings.device = device
yt_model.graph_embeddings.aggregator.device = device
rl_model = A2Clstm(env_args, video_embeddings)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)


with h5py.File(args.train_data_path, "r") as train_hf:
    train_inputs = np.array(train_hf["input"][:])
print(train_inputs.shape)


ray.init()
workers = [RolloutWorker.remote(env_args, train_inputs[i].tolist(), i) for i in range(env_args.num_browsers)]
env = Env(env_args, yt_model, rl_agent, workers, seed=0)

for i in range(train_inputs.shape[0] // env_args.num_browsers):
    for j in range(env_args.num_browsers):
        env.workers[j].user_videos = train_inputs[i * env_args.num_browsers + j].tolist()
    env.start_env()
    env.rollout()
    env.stop_env()
ray.shutdown()

