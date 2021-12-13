from denoiser import *
import json
import h5py
import torch
import torch.optim as optim
import logging

with open("../obfuscation/figs/dataset_final_joint_cate_100_2_test.json", "r") as json_file:
    data = json.load(json_file)

with open("../dataset/video_ids_final_joint_cate_100_2_test.json", "r") as json_file:
    video_ids = json.load(json_file)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

logging.basicConfig(
    filename=f"./logs/denoiser.txt",
    filemode='a',
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
        base_persona.append([video_ids[video] for video in data[f"rl_base_{i}"]["viewed"]])
        base_rec.append(data[f"rl_base_{i}"]["cate_dist"])

        obfu_persona.append([video_ids[video] for video in data[f"rl_obfu_{i}"]["viewed"]])
        obfu_rec.append(data[f"rl_obfu_{i}"]["cate_dist"])

        base_persona.append([video_ids[video] for video in data[f"rand_base_{i}"]["viewed"]])
        base_rec.append(data[f"rand_base_{i}"]["cate_dist"])

        obfu_persona.append([video_ids[video] for video in data[f"rand_obfu_{i}"]["viewed"]])
        obfu_rec.append(data[f"rand_obfu_{i}"]["cate_dist"])
    except:
        continue

train_dataloader, test_dataloader = get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=50, max_len=50)

with h5py.File("../dataset/video_embeddings_final_joint_cate_100_2_test_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")
video_embeddings = torch.from_numpy(video_embeddings).to(device)

# Define denoiser
denoiser_model = DenoiserNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device)
denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)
denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)

denoiser.train(train_dataloader)