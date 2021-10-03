import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np

USE_RAND = 2

with open("../dataset/home_video_id_sorted.json", "r") as json_file:
    home_video_id_sorted = json.load(json_file)
home_video_id_sorted = [int(key) for key in home_video_id_sorted.keys()]


class PolicyNetClassifier(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, num_videos=127085, device="cpu", use_rand=-1, topk=-1):
        super(PolicyNetClassifier,self).__init__()

        self.device = device
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        ).to(self.device)
        self.linear = nn.Linear(hidden_dim, num_videos).to(self.device)
        self.video_embeddings = video_embeddings # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.relu = nn.ReLU().to(self.device)
        self.use_rand = use_rand
        self.topk = topk
        self.num_videos = num_videos
        self.train()

    def forward(self, inputs, label, label_type, mask, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.video_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)
        
        label = label.to(self.device)
        label_type = label_type.to(self.device)
        mask = mask.to(self.device)
        
        last_label = []
        last_label_type = []
        for i in range(len(mask)):
            last_label.append(label[i][sum(mask[i]) - 1])
            last_label_type.append(label_type[i][sum(mask[i]) - 1])
        
        last_label = torch.stack(last_label, dim=0).squeeze().long()
        last_label_type = torch.stack(last_label_type, dim=0).squeeze().long()
        print(last_label.shape)
        batch_size, label_len = last_label.shape

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        out, _ = self.lstm(inputs)
        
        outputs = []
        for i in range(len(mask)):
            outputs.append(out[i][sum(mask[i]) - 1])
        outputs = torch.stack(outputs, dim=0).squeeze()

        # batch_size * seq_len * (2*hidden_dim) -> batch_size * seq_len * num_videos
        outputs = self.relu(self.linear(outputs))
        logits = torch.gather(F.log_softmax(outputs, -1), -1, last_label)
        
        # Get softmax and logits
        logits = last_label_type * logits
        logits = logits.mean(1)
        
        # Calculate different metrics
        # _, rec_rank_all = torch.sort(F.softmax(outputs, -1), descending=True, dim=-1) #torch.topk(F.softmax(outputs, -1), k=self.num_videos, dim=-1)
        # print(rec_rank_all.size())
        if self.topk == -1:
            _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=100, dim=-1) # rec_rank_all[:,:,:100]
        else:
            _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=self.topk, dim=-1) # rec_rank_all[:,:,:self.topk]
        
        # print(rec_outputs)
        if self.use_rand == 0:
            rec_outputs = np.random.choice(home_video_id_sorted, rec_outputs.size())
        elif self.use_rand == 1:
            rec_outputs = np.random.choice(home_video_id_sorted[:self.topk], rec_outputs.size())
        else:
            rec_outputs = rec_outputs.tolist()
            
        last_label = last_label.tolist()
        last_label_type = last_label_type.tolist()
        
        last_acc = 0
        last_count = 0
        last_avg_rank = 0

        for i in range(batch_size):
            num_rec = sum(last_label_type[i])
            if self.topk == -1:
                label_map = dict(zip(rec_outputs[i][:num_rec], [0 for _ in range(num_rec)]))
            else:
                label_map = dict(zip(rec_outputs[i], [0 for _ in range(len(rec_outputs[i]))]))

            for j in range(num_rec):
                if last_label[i][j] in label_map.keys():
                    label_map[last_label[i][j]] += 1
                    last_acc += 1
            last_count += num_rec

        return -logits, last_acc/last_count, last_count, last_acc/last_count, last_count, last_avg_rank/last_count
