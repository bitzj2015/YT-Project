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


with h5py.File(args.video_emb_path, "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")

yt_model = torch.load(
    "/scratch/param/policy_no_graph_edge_reg_new73_100_out_64_001_atten_new_aug.pt", 
    map_location=torch.device('cpu')
)
env_args = EnvConfig(action_dim=video_embeddings.shape[0])
rl_model = A2Clstm(env_args, video_embeddings)
rl_optimizer = optim.Adam(rl_model.parameters(), lr=env_args.rl_lr)
rl_agent = Agent(rl_model, rl_optimizer, env_args)

with h5py.File(args.train_data_path, "r") as train_hf:
    train_inputs = np.array(train_hf["input"][:])
print(train_inputs.shape)

workers = [RolloutWorker(env_args, train_inputs.tolist(), i) for i in range(env_args.num_browsers)]
env = Env(env_args, yt_model, rl_agent, workers, seed=0)

env.start_env()
env.rollout()
env.stop_env()

