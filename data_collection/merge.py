import os
import json
from tqdm import tqdm

puppet = [
{
    "type": "header",
    "version": "4.9.5deb2",
    "comment": "Export to JSON plugin for PHPMyAdmin"
},
{
    "type": "database",
    "name": "youtube"
},
{
    "type": "table",
    "name": "sock-puppets",
    "database": "youtube",
    "data": []
}
]

root_dir = "../../YT-Driver/docker-volume/crawls_new"
for user_dir in tqdm(os.listdir(root_dir)):
    try:
        for filename in os.listdir(f"{root_dir}/{user_dir}"):
            if filename.startswith("trail"):
                with open(f"{root_dir}/{user_dir}/{filename}") as json_file:
                    data = json.load(json_file)
                puppet[2]["data"].append(data)
    except:
        continue

print("Totoal number of valid sock puppets: {}".format(len(puppet[2]["data"])))
with open("../dataset/sock_puppets_final.json", "w") as json_file:
    json.dump(puppet, json_file)