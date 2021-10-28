import json
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.5
with open(f"./results/results/train_log_{ALPHA}.json", "r") as json_file:
    data = json.load(json_file)["reward"]

avg_data = []
print(len(data))
batch = 110
for i in range(len(data) // batch):
    avg_data.append(np.mean(data[i * batch: (i + 1) * batch]))
plt.figure()
plt.plot(avg_data)
plt.xlabel("epoch")
plt.ylabel("average reward")
plt.savefig(f"./results/results/train_{ALPHA}.png")