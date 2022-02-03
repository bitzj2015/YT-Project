from robust import *
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
    filename=f"./logs/robust_{args.version}.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


obfu_persona = []
obfu_rec = []
method = "rl"

for i in range(1500):
    try:
        base_videos = {video_ids[video]:0 for video in data[f"{method}_base_{i}"]["viewed"]}
        obfu_videos = []
        obfu_video_labels = []
        for video in data[f"{method}_obfu_{i}"]["viewed"]:
            obfu_videos.append(video_ids[video])
            if video_ids[video] in base_videos.keys():
                obfu_video_labels.append([1,0])
                del base_videos[video_ids[video]]
            else:
                obfu_video_labels.append([0,1])
        obfu_persona.append(obfu_videos)
        obfu_rec.append(obfu_video_labels)
    except:
        continue


train_dataloader, test_dataloader = get_robust_dataset(obfu_persona, obfu_rec, batch_size=50, max_len=50)

with h5py.File(f"../dataset/video_embeddings_{tag}_aug.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:].astype("float32")
video_embeddings = torch.from_numpy(video_embeddings).to(device)

# Define robust
robust_model = RobustNet(emb_dim=video_embeddings.shape[1], hidden_dim=256, video_embeddings=video_embeddings, device=device, base=args.use_base)
robust_optimizer = optim.Adam(robust_model.parameters(), lr=0.001)
robust = Robust(robust_model=robust_model, optimizer=robust_optimizer, logger=logger)

best_metric = 0
best_ep = 0
for ep in range(30):
    logger.info(f"Training epoch: {ep}")
    robust.train(train_dataloader)
    logger.info(f"Testing epoch: {ep}")
    _, _, f1= robust.eval(test_dataloader)
    if f1 > best_metric:
        best_metric = f1
        best_ep = ep
        torch.save(robust.robust_model, f"./param/robust_{args.version}.pkl")
print(best_metric, ep)
