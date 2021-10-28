import json
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.2
VERSION = "all"
with open(f"./results/train_log_{ALPHA}_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)["reward"]

avg_data = []
print(len(data))
batch = 110
for i in range(len(data) // batch):
    avg_data.append(np.mean(data[i * batch: (i + 1) * batch]))
plt.figure()
plt.plot(avg_data)
plt.xlabel("Epoch")
plt.ylabel("Average accumulative reward (within [0,2])")
plt.savefig(f"./results/train_{ALPHA}_{VERSION}.png")
