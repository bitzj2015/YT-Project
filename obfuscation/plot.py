import json
import matplotlib.pyplot as plt

with open("./results/train_log_0.5.json", "r") as json_file:
    data = json.load(json_file)
plt.figure()
plt.plot(data["reward"])
plt.savefig("./results/train_0.5.png")