import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *

class PolicyNetRegression(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, graph_embeddings, device="cpu"):
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
        self.linear = nn.Linear(hidden_dim, emb_dim).to(self.device)
        self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.train()

    def forward(self, inputs, label, label_type, mask, topk=-1, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)
        inputs = inputs * mask

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        outputs, _ = self.lstm(inputs)

        # batch_size * seq_len * hidden_dim -> batch_size * seq_len * 1 * emb_dim
        outputs = torch.tanh(self.linear(outputs)).unsqueeze(2)

        # batch_size * seq_len * emb_dim * num_videos
        label = label.long()
        batch_size, seq_len, label_len = label.shape
        sample_embedding = self.video_embeddings[label.reshape(-1).tolist()].reshape(batch_size, seq_len, label_len, self.emb_dim)
        sample_embedding = sample_embedding.permute(0,1,3,2)

        # (batch_size * seq_len * 1 * emb_dim) * (batch_size * seq_len * emb_dim * label_len) -> batch_size * seq_len * 1 * label_len
        logits = torch.matmul(outputs, sample_embedding) / ((torch.norm(outputs, dim=3).unsqueeze(3) * torch.norm(sample_embedding, dim=2).unsqueeze(2)) + 1e-10)
        logits = logits.squeeze(2)

        outputs = torch.matmul(outputs, self.video_embeddings.t()).squeeze(2)
        
        # Get accuracy
        if topk == -1:
            _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=100, dim=-1)
        else:
            _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=topk, dim=-1)
        
        if USE_RAND == 0:
            rec_outputs = np.random.choice(home_video_id_sorted, rec_outputs.size())
        elif USE_RAND == 1:
            rec_outputs = np.random.choice(home_video_id_sorted[:500], rec_outputs.size())
        else:
            rec_outputs = rec_outputs.tolist()

        mask = mask.squeeze(2)
        label = label.tolist()
        count = 0
        acc = 0
        last_acc = 0
        last_count = 0
        for i in range(len(mask)):
            for j in range(sum(mask[i])):
                num_rec = sum(label_type[i][j])
                if topk == -1:
                    label_map = dict(zip(rec_outputs[i][j][:num_rec], [0 for _ in range(num_rec)]))
                else:
                    label_map = dict(zip(rec_outputs[i][j], [0 for _ in range(len(rec_outputs[i][j]))]))
                for k in range(num_rec):
                    if label[i][j][k] in label_map.keys():
                        acc += 1
                        label_map[label[i][j][k]] += 1
                        if j == sum(mask[i]) - 1:
                            last_acc += 1

                count += num_rec
                if j == sum(mask[i]) - 1:
                    last_count += num_rec
                    # print("hhh")
                    # print(label_map, num_rec)
                    # print(label[i][j][:num_rec],rec_outputs[i][j][:num_rec])
        
        for i in range(len(mask)):
            for j in range(sum(mask[i]) - 1):
                mask[i][j] *= 0
        mask = mask.unsqueeze(2)
        
        # Get softmax and logits
        loss_neg = mask * (1-label_type) * logits
        loss_neg = loss_neg.sum(2) / ((1-label_type).sum(2) + 1e-10)
        loss_neg = loss_neg.sum(1)

        loss_pos = mask * label_type * (1-logits)
        loss_pos = loss_pos.sum(2) / (label_type.sum(2) + 1e-10)
        loss_pos = loss_pos.sum(1)
        
        return loss_pos, loss_neg, last_acc/last_count, last_count