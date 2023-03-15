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
parser.add_argument('--alpha', dest="alpha", type=str, default="0.2")
parser.add_argument('--use-base', dest="use_base", default=False, action='store_true')
args = parser.parse_args()

# tag = "final_joint_cate_100_2_test"
# tag = "final_joint_cate_103_2_test"
# tag = "realuser_0.2_test"
# tag_base = "realuser"
# tag = "latest_joint_cate_010_reddit3_0.2"
# tag_base = "reddit_40"
# tag = "latest_joint_cate_010"
# tag_base = "40"
# tag = "v1_binary_0.2_test"
# tag = "0.2_v2_kldiv_0.2_test_0.2_test"
ALPHA = args.version.split("_")[0]
tag = f"{ALPHA}_v2_kldiv_{ALPHA}_test"
tag = "0.3_v2_kldiv_pbooster_0.3_3_new"
tag = "0.2_v2_kldiv_pbooster_reddit_0.2_3_new_v2"
tag_base = "reddit_40_new"
# tag_base = "40_June"

# tag = f"{ALPHA}_v2_kldiv_reddit2_test"
# tag_base = "reddit_40_new"
# tag = "realuser_all"
# tag_base = "realuser_all"


with open(f"../obfuscation/figs/dataset_{tag}.json", "r") as json_file:
    data = json.load(json_file)

with open(f"/scratch/YT_dataset/dataset/video_ids_{tag_base}.json", "r") as json_file:
    video_ids = json.load(json_file)

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
TYPE = args.version.split("_")[1]
for i in range(1050):
    # try:
    base_persona.append([video_ids[video] for video in data[f"{TYPE}_base_{i}"]["viewed"]])
    base_rec.append(data[f"{TYPE}_base_{i}"]["cate_dist"])

    obfu_persona.append([video_ids[video] for video in data[f"{TYPE}_obfu_{i}"]["viewed"]])
    obfu_rec.append(data[f"{TYPE}_obfu_{i}"]["cate_dist"])
    if len(obfu_persona[-1]) > MAX_LEN:
        MAX_LEN = len(obfu_persona[-1])

        # base_persona.append([video_ids[video] for video in data[f"rand_base_{i}"]["viewed"]])
        # base_rec.append(data[f"rand_base_{i}"]["cate_dist"])

        # obfu_persona.append([video_ids[video] for video in data[f"rand_obfu_{i}"]["viewed"]])
        # obfu_rec.append(data[f"rand_obfu_{i}"]["cate_dist"])
    # except:
    #     continue

print(len(base_persona))

with h5py.File(f"/scratch/YT_dataset/dataset/video_embeddings_{tag_base}_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")
video_embeddings = torch.from_numpy(video_embeddings).to(device)

EVAL = False
kl_all = []
for t in range(3):
    # Define denoiser
    denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base)
    denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)
    denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)
    train_dataloader, val_dataloader, test_dataloader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=32, max_len=MAX_LEN)

    if not EVAL:
        best_kl = 10
        for ep in range(20):
            logger.info(f"Training epoch: {ep}")
            denoiser.train(train_dataloader)
            logger.info(f"Testing epoch: {ep}")
            _, kl = denoiser.eval(val_dataloader)
            if kl < best_kl:
                best_kl = kl
                torch.save(denoiser.denoiser_model, f"./param/denoiser_{args.version}.pkl")
        _, kl = denoiser.eval(test_dataloader)
        logger.info(f"Trial: {t}, kl: {kl}")
    else:
        denoiser.denoiser_model = torch.load(f"./param/denoiser_{args.version}.pkl")
        _, kl = denoiser.eval(test_dataloader)
        logger.info(f"Trial: {t}, kl: {kl}")
    kl_all.append(kl)

logger.info(f"Average kl: {np.mean(kl_all)}")
