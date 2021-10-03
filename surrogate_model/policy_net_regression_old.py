import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *

class PolicyNetRegression(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, graph_embeddings, video_embeddings, device="cpu", topk=-1, use_rand=-1):
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
        self.linear = nn.Linear(hidden_dim, emb_dim * 10).to(self.device)
        self.header1 = nn.Linear(emb_dim, 1).to(self.device)
        self.header2 = nn.Linear(emb_dim, 1).to(self.device)
        self.header = nn.Linear(10, 1).to(self.device)
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

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        outputs, _ = self.lstm(inputs)

        # batch_size * seq_len * hidden_dim -> batch_size * seq_len * 1 * emb_dim
        outputs = self.tanh(self.linear(outputs)).reshape(batch_size, seq_len, 10, -1)


        # sample_embedding = self.video_embeddings[label.reshape(-1).tolist()].reshape(batch_size, seq_len, label_len, self.emb_dim)
        # sample_embedding = sample_embedding.permute(0,1,3,2)
        # print(outputs.size(), sample_embedding.size(), sample_embedding.requires_grad)
        # (batch_size * seq_len * 1 * emb_dim) * (batch_size * seq_len * emb_dim * label_len) -> batch_size * seq_len * 1 * label_len
        # logits = torch.matmul(outputs, sample_embedding) / ((torch.norm(outputs, dim=3).unsqueeze(3) * torch.norm(sample_embedding, dim=2).unsqueeze(2)) + 1e-10)
        # logits = logits.squeeze(2)
        # emb_header1 = self.header1(self.video_embeddings).reshape(1, 1, -1, 1)
        # emb_header2 = self.header2(outputs)

        # emb_header = emb_header1 + emb_header2

        # outputs = torch.sigmoid(emb_header).squeeze(3)

        outputs = torch.matmul(outputs, self.video_embeddings.t()) # / ((torch.norm(outputs, dim=3).unsqueeze(3) * self.vb_norm) + 1e-10)
        outputs, _ = outputs.max(2)
        print(outputs.size())
        # logits = torch.gather(outputs, -1, label)
        
        # # Get accuracy
        # if topk == -1:
        #     _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=100, dim=-1)
        # else:
        #     _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=topk, dim=-1)
        
        # if USE_RAND == 0:
        #     rec_outputs = np.random.choice(home_video_id_sorted, rec_outputs.size())
        # elif USE_RAND == 1:
        #     rec_outputs = np.random.choice(home_video_id_sorted[:500], rec_outputs.size())
        # else:
        #     rec_outputs = rec_outputs.tolist()

        # mask = mask.squeeze(2)
        # label = label.tolist()
        # count = 0
        # acc = 0
        # last_acc = 0
        # last_count = 0
        # for i in range(len(mask)):
        #     for j in range(sum(mask[i])):
        #         num_rec = sum(label_type[i][j])
        #         if topk == -1:
        #             label_map = dict(zip(rec_outputs[i][j][:num_rec], [0 for _ in range(num_rec)]))
        #         else:
        #             label_map = dict(zip(rec_outputs[i][j], [0 for _ in range(len(rec_outputs[i][j]))]))
        #         for k in range(num_rec):
        #             if label[i][j][k] in label_map.keys():
        #                 acc += 1
        #                 label_map[label[i][j][k]] += 1
        #                 if j == sum(mask[i]) - 1:
        #                     last_acc += 1

        #         count += num_rec
        #         if j == sum(mask[i]) - 1:
        #             last_count += num_rec
        #             # print("hhh")
        #             # print(label_map, num_rec)
        #             # print(label[i][j][:num_rec],rec_outputs[i][j][:num_rec])
        
        # logits_mask  = mask.clone()
        # for i in range(len(mask)):
        #     for j in range(sum(mask[i]) - 1):
        #         logits_mask[i][j] *= 0

        # logits_mask = logits_mask.unsqueeze(2)
        
        # # Get softmax and logits
        # loss_neg = logits_mask * outputs
        # loss_neg = loss_neg.mean(2)
        # loss_neg = loss_neg.sum(1)

        # loss_pos = logits_mask * label_type * (1-logits)
        # loss_pos = loss_pos.sum(2) / (label_type.sum(2) + 1e-10)
        # loss_pos = loss_pos.sum(1)
        
        # return loss_pos, loss_neg, last_acc/last_count, last_count
        
        logits = torch.gather(F.log_softmax(outputs, -1), -1, label)

        for i in range(len(mask)):
            for j in range(sum(mask[i]) - 1):
                label_type[i][j] *= 0
                # label_type[i][j][0] = 1
        # logits = torch.gather(F.log_softmax(outputs, -1), -1, label)
        
        # Get softmax and logits
        logits = mask * label_type * logits
        logits = logits.mean(2) # / (label_type.sum(2) + 1e-10)
        logits = logits.sum(1)
        
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
            
        label = label.tolist()
        label_type = label_type.tolist()
        mask = mask.squeeze(2).tolist()
        count = 0
        acc = 0
        last_acc = 0
        last_count = 0
        last_avg_rank = 0

        for i in range(len(mask)):
            for j in range(sum(mask[i])):
                num_rec = sum(label_type[i][j])
                if self.topk == -1:
                    label_map = dict(zip(rec_outputs[i][j][:num_rec], [0 for _ in range(num_rec)]))
                else:
                    label_map = dict(zip(rec_outputs[i][j], [0 for _ in range(len(rec_outputs[i][j]))]))
                # rank_map = dict(zip(rec_rank_all[i][j].tolist(), [k for k in range(len(rec_rank_all[i][j]))]))
                
                for k in range(num_rec):
                    # if j == sum(mask[i]) - 1:
                    #     last_avg_rank += rank_map[label[i][j][k]]
                    if label[i][j][k] in label_map.keys():
                        acc += 1
                        label_map[label[i][j][k]] += 1
                        if j == sum(mask[i]) - 1:
                            last_acc += 1
                            # print(rec_outputs[i][j][k], inputs[i][j])
                            
                count += num_rec
                if j == sum(mask[i]) - 1:
                    last_count += num_rec
                    # print("check model output")
                    # print(label_map)
                    # print(label[i][j][:num_rec],rec_outputs[i][j][:num_rec])
            # break

        return -logits * 0.5, -logits * 0.5, last_acc/last_count, last_count