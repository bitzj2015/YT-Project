import os
import json
from pickle import TRUE
from tqdm import tqdm
root_path = "/scratch/YT_dataset"

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

# VERSION = "rl_reddit_new2"
# VERSION = "final_joint_cate_100_2_test"
# VERSION = "final_with_graph"
VERSION = "final_joint_cate_100_2_0.1"
VERSION = "final_joint_cate_103_2_test"
VERSION = "reddit_40"
VERSION = "latest_joint_cate_010_0.3"
# VERSION = "reddit_40_new"
# VERSION = "latest_joint_cate_010_reddit3_0.2"
VERSION = "v1_binary_0.2_test"
VERSION = "realuser_0.2_test_v2"
VERSION = "0.2_v2_kldiv_0.2_test"
VERSION = "realuser"

root_dir = f"./docker-volume/crawls_{VERSION}"
LOAD = False
if not LOAD:
    for user_dir in sorted(tqdm(os.listdir(root_dir))):
        try:
            for filename in os.listdir(f"{root_dir}/{user_dir}"):
                if filename.startswith("trail"):
                    with open(f"{root_dir}/{user_dir}/{filename}") as json_file:
                        data = json.load(json_file)
                    data["user_id"] = user_dir
                    puppet[2]["data"].append(data)
        except:
            continue
    print(data.keys())
    print("Totoal number of valid sock puppets: {}".format(len(puppet[2]["data"])))
    with open(f"{root_path}/dataset/sock_puppets_{VERSION}.json", "w") as json_file:
        json.dump(puppet, json_file)

else:
    for user_dir in sorted(tqdm(os.listdir(root_dir))):
        try:
            reload_home_rec = []
            init_home_rec = []
            for filename in os.listdir(f"{root_dir}/{user_dir}"):
                if filename.startswith("trail"):
                    with open(f"{root_dir}/{user_dir}/{filename}") as json_file:
                        data = json.load(json_file)
                    data["user_id"] = user_dir
                    puppet[2]["data"].append(data)

                elif filename.startswith("reload"):
                    with open(f"{root_dir}/{user_dir}/{filename}") as json_file:
                        data = json.load(json_file)
                    reload_home_rec = data["homepage"]
                    init_home_rec = data["initial_homepage"]
            puppet[2]["data"][-1]["homepage"] = reload_home_rec
            puppet[2]["data"][-1]["initial_homepage"] = init_home_rec
            assert puppet[2]["data"][-1]["user_id"] == user_dir
        except:
            continue
    print(data.keys())
    print("Totoal number of valid sock puppets: {}".format(len(puppet[2]["data"])))
    with open(f"{root_path}/dataset/sock_puppets_{VERSION}_reload.json", "w") as json_file:
        json.dump(puppet, json_file)