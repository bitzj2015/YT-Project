from ftplib import all_errors
from pickle import TRUE
from tkinter.tix import MAX
from denoiser import *
import json
import h5py
import torch
import torch.optim as optim
import logging
import argparse
import numpy as np
torch.random.manual_seed(1024)

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--version', dest="version", type=str, default="base3")
parser.add_argument('--use-base', dest="use_base", default=False, action='store_true')
args = parser.parse_args()

tag = "realuser"
tag_base = "40_June"
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

with open(f"../obfuscation/figs/dataset_{tag}.json", "r") as json_file:
    data = json.load(json_file)

with open(f"/scratch/YT_dataset/dataset/video_ids_{tag}.json", "r") as json_file:
    VIDEO_IDS = json.load(json_file)

# Load video embeddings
with h5py.File(f"/scratch/YT_dataset/dataset/video_embeddings_{tag}_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")

with h5py.File(f"/scratch/YT_dataset/dataset/video_embeddings_{tag_base}_aug.hdf5", "r") as hf_emb:
    video_embeddings_aug = hf_emb["embeddings"][:].astype("float32")

video_embeddings = np.concatenate([video_embeddings, video_embeddings_aug], axis=0)
video_embeddings = torch.from_numpy(video_embeddings).to(device)

with open(f"/scratch/YT_dataset/dataset/video_ids_40_June.json", "r") as json_file:
    VIDEO_IDS_AUG = json.load(json_file)

print(VIDEO_IDS["zYLVpPgGrec"])
AUG_LEN = len(VIDEO_IDS.keys())
print(AUG_LEN, len(VIDEO_IDS_AUG))
ID2VIDEO = {}
for key in VIDEO_IDS_AUG.keys():
    if key in VIDEO_IDS.keys():
        ID2VIDEO[str(VIDEO_IDS_AUG[key] + AUG_LEN)] = key
        continue
    else:
        VIDEO_IDS[key] = VIDEO_IDS_AUG[key] + AUG_LEN

print(video_embeddings.size())

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

logging.basicConfig(
    filename=f"./logs/denoiser_{args.version}.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 

print(len(data))

base_persona = []
obfu_persona = []
base_rec = []
obfu_rec = []
MAX_LEN = 0
TYPE = args.version.split("_")[-1]
for i in range(200):
    # try:
    base_persona.append([VIDEO_IDS[video] for video in data[f"rand_base_{i}"]["viewed"]])
    base_rec.append(data[f"rand_base_{i}"]["cate_dist"])

    obfu_persona.append([VIDEO_IDS[video] for video in data[f"{TYPE}_obfu_{i}"]["viewed"]])
    obfu_rec.append(data[f"{TYPE}_obfu_{i}"]["cate_dist"])
    if len(obfu_persona[-1]) > MAX_LEN:
        MAX_LEN = len(obfu_persona[-1])

    # base_persona.append([video_ids[video] for video in data[f"rand_base_{i}"]["viewed"]])
    # base_rec.append(data[f"rand_base_{i}"]["cate_dist"])

    # obfu_persona.append([video_ids[video] for video in data[f"rand_obfu_{i}"]["viewed"]])
    # obfu_rec.append(data[f"rand_obfu_{i}"]["cate_dist"])
    # except:
    #     continue

# train_dataloader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=32, max_len=MAX_LEN, all_eval=True)

# # Define denoiser
# denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base)
# denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)
# denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)


# _, kl = denoiser.eval(train_dataloader)
# print(kl)



EVAL = True
kl_list = []
for t in range(3):
    # Define denoiser
    denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base)
    denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)
    denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)


    if not EVAL:
        train_dataloader, val_dataloader, test_dataloader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=32, max_len=MAX_LEN)
        best_kl = 10
        for ep in range(50):
            logger.info(f"Training epoch: {ep}")
            denoiser.train(train_dataloader)
            logger.info(f"Testing epoch: {ep}")
            _, kl = denoiser.eval(val_dataloader)
            if kl < best_kl:
                best_kl = kl
                # torch.save(denoiser.denoiser_model, f"./param/denoiser_{args.version}.pkl")
        _, kl = denoiser.eval(test_dataloader)
        logger.info(f"Trial: {t}, kl: {kl}")
    else:
        test_dataloader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=32, max_len=MAX_LEN, all_eval=True)
        denoiser.denoiser_model = torch.load(f"./param/denoiser_0.2_{TYPE}.pkl").to(device)
        denoiser.denoiser_model.video_embeddings = video_embeddings
        _, kl = denoiser.eval(test_dataloader)
        logger.info(f"Trial: {t}, kl: {kl}")
    kl_list.append(kl)

print(np.mean(kl_list))