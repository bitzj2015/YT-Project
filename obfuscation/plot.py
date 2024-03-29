import json
import numpy as np
import matplotlib.pyplot as plt

ALPHA = 0.2
VERSION = "final_joint_cate_100_2_0.3_0"
# VERSION = "reddit_new3_cate"
VERSION = "latest_joint_cate_010_0"
VERSION = "cate_40_June"
VERSION = "v1_binary"
# VERSION = "latest_joint_cate_010_reddit3_1"
VERSION = "realuser_2"
VERSION = "v2_kldiv_sensitive_new_50_v21_sens_0"
PHASE = "test"
with open(f"./results/{PHASE}_log_{ALPHA}_{VERSION}.json", "r") as json_file:
    data = json.load(json_file)["reward"]

avg_data = []
print(len(data), np.mean(data))
batch = 1
for i in range(len(data) // batch):
    avg_data.append(np.mean(data[i * batch: (i + 1) * batch]))
plt.figure()
plt.plot(avg_data)
plt.xlabel("Epoch")
plt.ylabel("Average accumulative reward (within [0,2])")
plt.savefig(f"./results/{PHASE}_{ALPHA}_{VERSION}.png")
