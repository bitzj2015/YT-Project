import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *

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
        self.linear2 = nn.Linear(hidden_dim, emb_dim * num_user_state).to(self.device)
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

        # batch_size * seq_len * emb_dim * num_videos
        label = label.long()
        batch_size, seq_len, label_len = label.shape

        last_label = []
        last_label_type = []
        for i in range(len(mask)):
            last_label.append(label[i][sum(mask[i]) - 1])
            last_label_type.append(label_type[i][sum(mask[i]) - 1])
        
        last_label = torch.stack(last_label, dim=0).squeeze().long()
        last_label_type = torch.stack(last_label_type, dim=0).squeeze().long()
        print(last_label.shape)

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        outputs, (last_hidden, _) = self.lstm(inputs)
        print(last_hidden.size())

        # batch_size * seq_len * hidden_dim -> batch_size * seq_len * 1 * emb_dim
        outputs = self.tanh(self.linear(outputs)).reshape(batch_size, seq_len, self.num_user_state, -1)
        outputs = torch.matmul(outputs, self.video_embeddings.t()) # / ((torch.norm(outputs, dim=3).unsqueeze(3) * self.vb_norm) + 1e-10)
        outputs, _ = outputs.max(2)
        
        last_outputs = self.tanh(self.linear2(last_hidden[:, -1, :])).reshape(batch_size, self.num_user_state, -1)
        last_outputs = torch.matmul(last_outputs, self.video_embeddings.t()) # / ((torch.norm(outputs, dim=3).unsqueeze(3) * self.vb_norm) + 1e-10)
        last_outputs, _ = last_outputs.max(1)
        
        logits = torch.gather(F.log_softmax(outputs, -1), -1, label)
        last_logits = torch.gather(F.log_softmax(last_outputs, -1), -1, last_label)
        
        # Get softmax and logits
        logits = mask * label_type * logits
        logits = logits.mean(2) # / (label_type.sum(2) + 1e-10)
        logits = logits.mean(1)
        
        last_logits = last_label_type * last_logits
        last_logits = last_logits.mean(1)
        
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

        if self.topk == -1:
            _, last_rec_outputs = torch.topk(F.softmax(last_outputs, -1), k=100, dim=-1) # rec_rank_all[:,:,:100]
        else:
            _, last_rec_outputs = torch.topk(F.softmax(last_outputs, -1), k=self.topk, dim=-1) # rec_rank_all[:,:,:self.topk]
        
        # print(rec_outputs)
        if self.use_rand == 0:
            last_rec_outputs = np.random.choice(home_video_id_sorted, last_rec_outputs.size())
        elif self.use_rand == 1:
            last_rec_outputs = np.random.choice(home_video_id_sorted[:self.topk], size=last_rec_outputs.size())#, p=home_video_value_sorted)
        else:
            last_rec_outputs = last_rec_outputs.tolist()

        label = label.tolist()
        label_type = label_type.tolist()
        mask = mask.squeeze(2).tolist()
        
        count = 0
        acc = 0
        last_acc = 0
        last_count = 0

        for i in range(len(mask)):
            for j in range(sum(mask[i])-1):
                num_rec = sum(label_type[i][j])
                if self.topk == -1:
                    label_map = dict(zip(rec_outputs[i][j][:num_rec], [0 for _ in range(num_rec)]))
                else:
                    label_map = dict(zip(rec_outputs[i][j], [0 for _ in range(len(rec_outputs[i][j]))]))

                for k in range(num_rec):
                    if label[i][j][k] in label_map.keys():
                        acc += 1
                        label_map[label[i][j][k]] += 1
                count += num_rec
                
            num_rec = sum(last_label_type[i])
            if self.topk == -1:
                label_map = dict(zip(last_rec_outputs[i][:num_rec], [0 for _ in range(num_rec)]))
            else:
                label_map = dict(zip(last_rec_outputs[i], [0 for _ in range(len(last_rec_outputs[i]))]))

            for j in range(num_rec):
                if last_label[i][j] in label_map.keys():
                    label_map[last_label[i][j]] += 1
                    last_acc += 1
            last_count += num_rec

        return -logits * 0.5, -last_logits * 0.5, last_acc/last_count, last_count, acc/count