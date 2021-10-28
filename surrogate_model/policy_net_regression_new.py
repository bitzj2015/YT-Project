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
        self.linear = nn.Linear(hidden_dim, emb_dim * 5).to(self.device)
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

    def forward(self, inputs, label, label_type, last_label, last_label_p, last_label_type, mask, topk=-1, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)

        label = label.to(self.device)
        label_type = label_type.to(self.device)
        last_label = last_label.to(self.device)
        last_label_p = last_label_p.to(self.device)
        last_label_type = last_label_type.to(self.device)
        mask = mask.to(self.device)

        # batch_size * seq_len * emb_dim * num_videos
        label = label.long()
        batch_size, seq_len, label_len = label.shape
        last_batch_size, last_label_len = last_label.shape
        print(label.shape)

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        out, (last_hidden, _) = self.lstm(inputs)
        print(last_hidden.size())

        sample_embedding = self.video_embeddings[label.reshape(-1).tolist()].reshape(batch_size, seq_len, label_len, self.emb_dim)
        sample_embedding = sample_embedding.permute(0,1,3,2)
        sample_last_embedding = self.video_embeddings[last_label.reshape(-1).tolist()].reshape(batch_size, last_label_len, self.emb_dim)
        sample_last_embedding = sample_last_embedding.permute(0,2,1)

        # batch_size * seq_len * hidden_dim -> batch_size * seq_len * 1 * emb_dim
        out = self.tanh(self.linear(out)).reshape(batch_size, seq_len, 5, -1)
        # outputs = torch.matmul(outputs, self.video_embeddings.t())
        outputs = torch.matmul(out, sample_embedding)
        outputs, _ = outputs.max(2)
        
        last_out = self.tanh(self.linear2(last_hidden[-1, :, :])).reshape(batch_size, self.num_user_state, -1)
        last_outputs = torch.matmul(last_out, sample_last_embedding)
        last_outputs, _ = last_outputs.max(1)
        
        last_outputs_all = torch.matmul(last_out, self.video_embeddings.t())
        last_outputs_all, _ = last_outputs_all.max(1)
        
        print(outputs.size(), last_outputs.size())
        
        '''
        logits = torch.gather(F.log_softmax(outputs, -1), -1, label)
        last_logits = torch.gather(F.log_softmax(last_outputs, -1), -1, last_label)
        
        # Get softmax and logits
        logits = mask * label_type * logits
        logits = logits.sum(2) / (label_type.sum(2) + 1e-10)
        logits = logits.mean(1)
        
        last_logits = last_label_type * last_logits
        last_logits = last_logits.sum(1) / (last_label_type.sum(1) + 1e-10)
        '''
        logits_all = F.log_softmax(outputs, -1)
        last_logits_all = F.log_softmax(last_outputs, -1)
        logits_pos = mask * label_type * logits_all
        # logits_neg = mask * (1 - label_type) * logits_all
        last_logits_pos = last_label_type * last_logits_all
        # last_logits_neg = (1 - last_label_type) * last_logits_all

        logits = logits_pos.sum(2) / (label_type.sum(2) + 1e-10) # + logits_neg.sum(2) / ((1 - label_type).sum(2) + 1e-10)
        logits = logits.mean(1)
        last_logits = last_logits_pos.sum(1) / (last_label_type.sum(1) + 1e-10) # + last_logits_neg.sum(1) / ((1 - last_label_type).sum(1) + 1e-10)

        _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=100, dim=-1) # rec_rank_all[:,:,:100]
        _, last_rec_outputs = torch.topk(F.softmax(last_outputs, -1), k=100, dim=-1) # rec_rank_all[:,:,:100]
        _, last_rec_outputs_all = torch.topk(F.softmax(last_outputs_all, -1), k=100, dim=-1) # rec_rank_all[:,:,:100]
        
        print(rec_outputs.size(), last_rec_outputs.size())
        rec_outputs = torch.gather(label, -1, rec_outputs)
        last_rec_outputs = torch.gather(last_label, -1, last_rec_outputs)

        rec_outputs = rec_outputs.tolist()
        last_rec_outputs = last_rec_outputs.tolist()
        last_rec_outputs_all = last_rec_outputs_all.tolist()

        label = label.tolist()
        label_type = label_type.tolist()
        last_label = last_label.tolist()
        last_label_type = last_label_type.tolist()
        mask = mask.squeeze(2).tolist()
        
        count = 0
        acc = 0
        last_acc = 0
        last_count = 0

        for i in range(len(mask)):
            for j in range(sum(mask[i])):
                num_rec = sum(label_type[i][j])
                label_map = dict(zip(rec_outputs[i][j][:num_rec], [0 for _ in range(num_rec)]))

                for k in range(num_rec):
                    if label[i][j][k] in label_map.keys():
                        acc += 1
                        label_map[label[i][j][k]] += 1
                count += num_rec
                
            num_rec = sum(last_label_type[i])
            label_map = dict(zip(last_rec_outputs_all[i][:num_rec], [0 for _ in range(num_rec)]))

            for j in range(num_rec):
                if last_label[i][j] in label_map.keys():
                    label_map[last_label[i][j]] += 1
                    last_acc += 1
                else:
                    for video_id in list(video_graph_adj_mat[str(last_label[i][j])].keys()):
                        if int(video_id) in label_map.keys():
                            label_map[int(video_id)] += 1
                            last_acc += 1
                            break
            last_count += num_rec

        return -logits * 0.0, -last_logits * 0.5, last_acc/last_count, last_count, acc/count, count