from stealthy import *
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

ALPHA = args.version.split("_")[0]
tag = f"{ALPHA}_v2_kldiv_{ALPHA}_test"
tag_base = "40_June"
# with open(f"../obfuscation/figs/dataset_{tag}.json", "r") as json_file:
#     data = json.load(json_file)

with open(f"/scratch/YT_dataset/dataset/sock_puppets_40_June.json", "r") as json_file:
    user_data = json.load(json_file)

with open(f"../obfuscation/results/test_user_trace_{tag}_0_0_new.json", "r") as json_file:
    rl_user_data = json.load(json_file)

with open(f"../obfuscation/results/test_user_trace_{tag}_1_1_new.json", "r") as json_file:
    rand_user_data = json.load(json_file)

with open(f"../obfuscation/results/test_user_trace_{tag}_2_2_new.json", "r") as json_file:
    bias_user_data = json.load(json_file)

# with open(f"../obfuscation/results/test_user_trace_0.5_v2_kldiv_pbooster_0.5_3_new.json", "r") as json_file:
#     rl_user_data = json.load(json_file)

# with open(f"../obfuscation/results/test_user_trace_0.3_v2_kldiv_pbooster_0.3_3_new.json", "r") as json_file:
#     rand_user_data = json.load(json_file)

# with open(f"../obfuscation/results/test_user_trace_0.2_v2_kldiv_pbooster_3_new.json", "r") as json_file:
#     bias_user_data = json.load(json_file)


data = {}
for i in range(1800):
    data[f"rl_base_{i}"] = rl_user_data["base"][str(i)]
    data[f"rl_obfu_{i}"] = rl_user_data["obfu"][str(i)]
    data[f"rand_base_{i}"] = rand_user_data["base"][str(i)]
    data[f"rand_obfu_{i}"] = rand_user_data["obfu"][str(i)]
    data[f"bias_obfu_{i}"] = bias_user_data["obfu"][str(i)]

with open(f"/scratch/YT_dataset/dataset/video_ids_{tag_base}.json", "r") as json_file:
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
persona = []

method = args.version.split("_")[1]
print(method)


with h5py.File(f"/scratch/YT_dataset/dataset/train_data_{tag_base}.hdf5", "r") as train_hf:
    train_inputs = np.array(train_hf["input"][:])

print(np.array(train_inputs).shape)

cnt = 0
for i in range(train_inputs.shape[0]):
    persona.append(list(train_inputs[i][-30:]))
    cnt += 1
    if cnt == 6840 / 19 * 19:
        break

for i in range(1800):
    try:
        base_persona.append([video_ids[video] for video in data[f"rand_base_{i}"]][-30:])
        base_len = len(base_persona[-1])
        obfu_persona.append([video_ids[video] for video in data[f"{method}_obfu_{i}"]][-base_len:])
    except:
        continue

print(len(base_persona))

print(len(obfu_persona))
train_dataloader, val_dataloader, test_dataloader = get_stealthy_dataset_v2(
    persona,
    base_persona, 
    obfu_persona, 
    batch_size=50, max_len=30)

with h5py.File(f"/scratch/YT_dataset/dataset/video_embeddings_{tag_base}_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")
video_embeddings = torch.from_numpy(video_embeddings).to(device)

# Define stealthy
stealthy_model = StealthyNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base).to(device)
stealthy_optimizer = optim.Adam(stealthy_model.parameters(), lr=0.001)
stealthy = Stealthy(stealthy_model=stealthy_model, optimizer=stealthy_optimizer, logger=logger)

best_metric = 0
best_acc = 0
best_f1 = [0,0,0]
for ep in range(30):
    logger.info(f"Training epoch: {ep}")
    stealthy.train(train_dataloader)
    logger.info(f"Testing epoch: {ep}")
    _, _, _, _, f1= stealthy.eval(val_dataloader)
    if f1 > best_metric:
        best_metric = f1
        best_ep = ep
        _, best_acc, best_f1[0], best_f1[1], best_f1[2] = stealthy.eval(test_dataloader)
        torch.save(stealthy.stealthy_model, f"./param/stealthy_{args.version}.pkl")
print(best_acc, best_ep, best_f1)
