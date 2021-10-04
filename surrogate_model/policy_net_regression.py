import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
import math

class PolicyNetRegression(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, graph_embeddings, video_embeddings, device="cpu", topk=-1, use_rand=-1, num_user_state=100):
        super(PolicyNetRegression,self).__init__()

        self.device = device
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        ).to(self.device)
        self.linear = nn.Linear(hidden_dim, emb_dim * num_user_state).to(self.device)
        self.num_user_state = num_user_state
        self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        self.vb_norm = torch.norm(self.video_embeddings, dim=1).reshape(1,1,1,-1)
        self.emb_dim = emb_dim
        self.topk = topk
        self.use_rand = use_rand
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.train()

    def forward(self, inputs, label, label_type, mask, topk=-1, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
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
        out, (last_hidden, _) = self.lstm(inputs)
        print(last_hidden.size())
        # outputs = last_hidden[-1]
        
        outputs = []
        for i in range(len(mask)):
            outputs.append(out[i][sum(mask[i]) - 1])
        outputs = torch.stack(outputs, dim=0).squeeze()
        

        # batch_size * hidden_dim -> batch_size * num_user_state * emb_dim
        outputs = self.linear(outputs).reshape(batch_size, self.num_user_state, -1)

        outputs = torch.matmul(outputs, self.video_embeddings.t()) # / ((torch.norm(outputs, dim=3).unsqueeze(3) * self.vb_norm) + 1e-10)
        outputs, _ = outputs.max(1)
        # outputs = outputs / math.sqrt(self.emb_dim)
        print(outputs.size())

        # atten = torch.matmul(outputs, self.video_embeddings.t()) # / ((torch.norm(outputs, dim=3).unsqueeze(3) * self.vb_norm) + 1e-10)
        # # # batch_size * num_user_state * num_videos
        # # print(atten.size())
        # # atten = F.softmax(atten, -2)
        # # print(atten.size())
        # # # batch_size * emb_dim * num_videos
        # # atten = torch.matmul(outputs.permute(0, 2, 1), atten)
        # # print(atten.size())
        # # print(self.video_embeddings.t().unsqueeze(0).size(), outputs.size())
        # # outputs = atten * (self.video_embeddings.t().unsqueeze(0))
        # # print(outputs.size())
        # # outputs = outputs.sum(1).squeeze()
        # # print(outputs.size())
        # outputs = atten * F.softmax(atten, -2)
        # outputs = outputs.sum(1).squeeze()

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

        return -logits * 0.5, -logits * 0.5, last_acc/last_count, last_count