import os
import h5py
import json
import torch
import numpy as np
from Social_Encoders import *
from Social_Aggregators import *
from Policy_Net import *
from torch.utils.data import DataLoader
from dataset import YTDataset, ToTensor
import torch.optim as optim
import logging

logging.basicConfig(
    filename="./logs/log_train3.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 

VERSION = "_new"

hf_emb = h5py.File(f"../dataset/video_embeddings{VERSION}.hdf5", "r")
video_embeddings = hf_emb["embeddings"][:]
video_ids = hf_emb["video_ids"][:]
video_ids = [video_id.decode('utf-8') for video_id in video_ids]

with open("../dataset/video_adj_list.json", "r") as json_file:
    video_graph_adj_mat = json.load(json_file)
num_videos = len(video_graph_adj_mat.keys())
print("No. of videos: {}".format(num_videos))

with open(f"../dataset/video_ids{VERSION}.json", "r") as json_file:
    video_ids_map = json.load(json_file)

# print(video_ids_map[video_ids[1000]])

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

embed_dim = np.shape(video_embeddings)[1]
print("video embedding dimension: {}".format(embed_dim))

device = torch.device("cuda" if use_cuda else "cpu")
video_embeddings = torch.from_numpy(video_embeddings)
agg_video_graph = Social_Aggregator(
    video_embeddings=video_embeddings, 
    embed_dim=embed_dim, 
    device=device
)

video_graph_embeddings = Social_Encoder(
    video_embeddings=video_embeddings, 
    embed_dim=embed_dim, 
    video_graph_adj_mat=video_graph_adj_mat, 
    aggregator=agg_video_graph,
    device=device
)
# video_embeddings = video_graph_embeddings([i for i in range(num_videos)])

policy_net = PolicyNet(
    embed_dim=embed_dim,
    hidden_dim=12800,
    video_embeddings=video_graph_embeddings
)

train_hf = h5py.File(f"../dataset/train_data{VERSION}.hdf5", "r")
test_hf = h5py.File(f"../dataset/test_data{VERSION}.hdf5", "r")
batch_size = 256

train_dataset = YTDataset(train_hf["input"], train_hf["label"], train_hf["label_type"], train_hf["mask"], transform=ToTensor())
test_dataset = YTDataset(test_hf["input"], test_hf["label"], test_hf["label_type"], test_hf["mask"], transform=ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
for ep in range(30):
    for i, batch in enumerate(train_loader):
        input, label, label_type, mask = batch["input"], batch["label"], batch["label_type"], batch["mask"]
        # print(input.size(), label.size(), label_type.size(), mask.size())
        optimizer.zero_grad()
        loss, loss_, _ = policy_net(input,label,label_type,mask)
        loss = loss.mean(0)
        loss_ = loss_.mean(0)
        # print(loss)
        (loss+loss_).backward()
        optimizer.step()
        logger.info("Training, ep: {}, step: {}, loss: {}, loss_: {}.".format(ep, i, loss.item(), loss_.item()))
    
    policy_net_ = policy_net.eval()
    for i, batch in enumerate(test_loader):
        input, label, label_type, mask = batch["input"], batch["label"], batch["label_type"], batch["mask"]
        # print(input.size(), label.size(), label_type.size(), mask.size())
        loss, loss_, _ = policy_net_(input,label,label_type,mask)
        loss = loss.mean(0)
        loss_ = loss_.mean(0)
        logger.info("Testing, ep: {}, step: {}, loss: {}, loss_: {}.".format(ep, i, loss.item(), loss_.item()))

