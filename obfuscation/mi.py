import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import random
from sklearn.cluster import KMeans


class Mine(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=512):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, 1)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        torch.nn.init.normal_(self.fc1.weight,std=0.01)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.normal_(self.fc2.weight,std=0.01)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.normal_(self.fc3.weight,std=0.01)
        torch.nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.relu(self.bn(self.fc1(input)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et) + 1e-9)
    return mi_lb, t, et

def learn_mine(batch, device, mine_net, mine_net_optim, ma_et, ma_rate=0.05):
    # batch is a tuple of (joint, marginal)
    joint , marginal = batch["joint"].to(device), batch["margin"].to(device)
    # joint = torch.autograd.Variable(torch.FloatTensor(joint)).to(device)
    # marginal = torch.autograd.Variable(torch.FloatTensor(marginal)).to(device)
    mi_lb , t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)
    
    # unbiasing use moving average
    loss = -(torch.mean(t) - (1/(ma_et.mean() + 1e-9)).detach()*torch.mean(et))
    # use biased estimator
#     loss = - mi_lb
    
    mine_net_optim.zero_grad()
    torch.autograd.backward(loss)
    mine_net_optim.step()
    return mi_lb, ma_et



class TaskDataset(Dataset):
    def __init__(self, input, label, client_id=-1):
        self.input = input
        self.label = label
        self.client_id = client_id

    def __len__(self):
        return np.shape(self.label)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        x = self.input[idx]
        x = x.reshape(x.size(0), x.size(1), -1).permute(2, 0, 1)
        y = self.label[idx]
        sample = {'x': x, 'y': y}
        return sample


class MINEDataset(Dataset):
    def __init__(self, joint, margin):
        self.joint = joint
        self.margin = margin

    def __len__(self):
        return np.shape(self.joint)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        joint = self.joint[idx]
        margin = self.margin[idx]
        sample = {'joint': joint, 'margin': margin}
        return sample


# Define hyperparameters
VERSION = "0.3_v2_kldiv_0.3_test"
with open(f"./figs/{VERSION}_metadata.json","r") as json_file:
    cate_dist = json.load(json_file)

for TAG in ["rand_obfu", "bias_obfu", "rl_obfu", "rand_base"]:
    obf_dist_all = []
    org_dist_all = []
    all_ret = []
    cnt = 0
    for i in cate_dist.keys():
        obf_dist_all.append([v + 1e-9 for v in cate_dist[i][TAG]])
        org_dist_all.append([v + 1e-9 for v in cate_dist[i][f"rand_base"]])
        cnt += 1

    kmeans = KMeans(n_clusters=20).fit(np.array(org_dist_all))

    obf_dist_g = {}
    org_dist_g = {}
    for i in range(len(kmeans.labels_)):
        l = kmeans.labels_[i]
        if l not in obf_dist_g.keys():
            obf_dist_g[l] = []
            org_dist_g[l] = []

        obf_dist_g[l].append(obf_dist_all[i])
        org_dist_g[l].append(org_dist_all[i])
        

    for l in obf_dist_g.keys():
        obf_dist = obf_dist_g[l]
        org_dist = org_dist_g[l]

        num_iter = 5000
        use_cuda = False
        if torch.cuda.is_available():
            use_cuda = True
        device = torch.device("cuda" if use_cuda else "cpu")

        # Get joint distributino of X and Y
        X = np.array(obf_dist)
        # random.shuffle(org_dist)
        Y = np.array(org_dist)
        joint = torch.from_numpy(np.concatenate([X, Y], axis=1).astype("float32"))


        for t in range(2):
            # Define MINE network
            mine_net = Mine(input_size=len(obf_dist[-1]) * 2).to(device)
            mine_net_optim = torch.optim.Adam(mine_net.parameters(), lr=0.0001)
            mine_net.train()

            # Get marginal distribution of Y as Y_ 
            random.shuffle(org_dist)
            Y_ = np.array(org_dist)
            margin = torch.from_numpy(np.concatenate([X, Y_], axis=1).astype("float32"))

            # Define MINE dataset
            mine_dataset = MINEDataset(joint, margin)
            mine_traindataloader = DataLoader(mine_dataset, batch_size=len(obf_dist), shuffle=True)

            # Train MINE network
            ans = 0
            for niter in range(num_iter):
                mi_lb_sum = 0
                ma_et = 1
                for i, batch in enumerate(mine_traindataloader):
                    mi_lb, ma_et = learn_mine(batch, device, mine_net, mine_net_optim, ma_et)
                    mi_lb_sum += mi_lb
                if niter % 1000 == 0:
                    print(f"MINE iter: {niter}, MI estimation: {mi_lb_sum / (i+1)}")
                ans = max(ans, mi_lb_sum / (i+1))
            all_ret.append(ans.item())
        print(l, TAG, np.mean(all_ret))
