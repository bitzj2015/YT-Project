from stealthy import *
import json
import h5py
import torch
import torch.optim as optim
import logging
import argparse

# Define arguments for training script
parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--version', dest="version", type=str, default="base3")
parser.add_argument('--use-base', dest="use_base", default=False, action='store_true')
args = parser.parse_args()

tag = "final_joint_cate_100_2_test"

with open(f"../obfuscation/figs/dataset_{tag}.json", "r") as json_file:
    data = json.load(json_file)

with open(f"../dataset/video_ids_{tag}.json", "r") as json_file:
    video_ids = json.load(json_file)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

logging.basicConfig(
    filename=f"./logs/stealthy_{args.version}.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


base_persona = []
obfu_persona = []
base_rec = []
obfu_rec = []

for i in range(1500):
    try:
        base_persona.append([video_ids[video] for video in data[f"rl_base_{i}"]["viewed"]][0:30])
        base_rec.append(data[f"rl_base_{i}"]["cate_dist"])

        obfu_persona.append([video_ids[video] for video in data[f"rl_obfu_{i}"]["viewed"]][0:30])
        obfu_rec.append(data[f"rl_obfu_{i}"]["cate_dist"])
        # print(len(data[f"rl_base_{i}"]["viewed"]), len(data[f"rl_obfu_{i}"]["viewed"]))

        # base_persona.append([video_ids[video] for video in data[f"rand_base_{i}"]["viewed"]][0:30])
        # base_rec.append(data[f"rand_base_{i}"]["cate_dist"])

        # obfu_persona.append([video_ids[video] for video in data[f"rand_obfu_{i}"]["viewed"]][0:30])
        # obfu_rec.append(data[f"rand_obfu_{i}"]["cate_dist"])
    except:
        continue


train_dataloader, test_dataloader = get_stealthy_dataset(base_persona, obfu_persona, batch_size=50, max_len=50)

with h5py.File(f"../dataset/video_embeddings_{tag}_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")
video_embeddings = torch.from_numpy(video_embeddings).to(device)

# Define stealthy
stealthy_model = StealthyNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base)
stealthy_optimizer = optim.Adam(stealthy_model.parameters(), lr=0.001)
stealthy = Stealthy(stealthy_model=stealthy_model, optimizer=stealthy_optimizer, logger=logger)

best_kl = 0
for ep in range(30):
    logger.info(f"Training epoch: {ep}")
    stealthy.train(train_dataloader)
    logger.info(f"Testing epoch: {ep}")
    _, kl = stealthy.eval(test_dataloader)
    if kl > best_kl:
        best_kl = kl
        torch.save(stealthy.stealthy_model, f"./param/stealthy_{args.version}.pkl")
print(best_kl)
