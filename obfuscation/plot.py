import json
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.2
VERSION = "reddit_joint_cate_100_2"
# VERSION = "reddit_new3_cate"
PHASE = "train"
with open(f"./results/{PHASE}_log_{ALPHA}_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)["reward"]

avg_data = []
print(len(data), np.mean(data))
batch = 200
for i in range(len(data) // batch):
    avg_data.append(np.mean(data[i * batch: (i + 1) * batch]))
plt.figure()
plt.plot(avg_data)
plt.xlabel("Epoch")
plt.ylabel("Average accumulative reward (within [0,2])")
plt.savefig(f"./results/{PHASE}_{ALPHA}_{VERSION}.png")
