import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from constants import *
from sklearn.metrics import precision_recall_fscore_support


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
        self.linear = nn.Linear(hidden_dim, 154).to(self.device)
        # self.linear2 = nn.Linear(emb_dim, emb_dim).to(self.device)
        self.num_user_state = num_user_state
        self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        # self.vb_norm = torch.norm(self.video_embeddings, dim=1).reshape(1,1,1,-1)
        self.emb_dim = emb_dim
        self.topk = topk
        self.use_rand = use_rand
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.train()
        self.w = torch.Tensor(CLASS_WEIGHT).to(self.device).reshape(1, -1)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def get_rec(self, inputs, with_graph=False, topk=100):
        batch_size, seq_len = inputs.shape
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        out, _ = self.lstm(inputs)
        outputs = []
        for i in range(batch_size):
            outputs.append(out[i][-1])
        outputs = torch.stack(outputs, dim=0).squeeze()
        # outputs = self.relu(self.linear(outputs))
        # outputs = F.softmax(outputs, -1)
        # emb = outputs.detach()
        outputs = torch.sigmoid(self.linear(outputs))
        emb = (outputs >= 0.5).int()
         
        return emb.cpu().numpy()

    def forward(self, inputs, label, mask, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)

        last_label = label.to(self.device)
        # last_label_type = label_type.to(self.device)
        mask = mask.to(self.device)

        batch_size, _ = last_label.shape

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        out, _ = self.lstm(inputs)
        # print(last_hidden.size())
        # outputs = last_hidden[-1]
        
        outputs = []
        for i in range(len(mask)):
            outputs.append(out[i][sum(mask[i]) - 1])
        outputs = torch.stack(outputs, dim=0).squeeze()
        
        # batch_size * hidden_dim -> batch_size * num_classes
        # outputs = self.relu(self.linear(outputs))
        # outputs = F.log_softmax(outputs, -1)
        # loss = self.kl_loss(outputs, label)
        # # print(self.kl_loss(torch.log(label[:-10] + 1e-7), label[10:]))

        # return loss, 0, batch_size, 0, 0, 0

        
        outputs = torch.sigmoid(self.linear(outputs))
        loss = label * torch.log(outputs) + (1 - label) * torch.log(1 - outputs)
        loss = -loss.mean()
        pred = (outputs >= 0.5).int()
        acc = (pred == label).sum() / (label.size(0) * label.size(1))
        haming_dis = torch.abs(pred - label).sum(-1) / 20
        precision, recall, f1, _ = precision_recall_fscore_support(pred.reshape(-1).numpy(), label.reshape(-1).numpy())
        print(haming_dis.mean())

        return loss, acc, batch_size, f1[1], precision[1], recall[1]

        '''
    def forward(self, inputs, label, label_cate, label_type, mask, with_graph=False):

        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.graph_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)

        last_label = label.to(self.device)
        last_cate = label_cate.to(self.device)
        # last_label_type = label_type.to(self.device)
        mask = mask.to(self.device)

        batch_size, _ = last_label.shape

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * hidden_dim
        out, _ = self.lstm(inputs)
        # print(last_hidden.size())
        # outputs = last_hidden[-1]
        
        outputs = []
        for i in range(len(mask)):
            outputs.append(out[i][sum(mask[i]) - 1])
        outputs = torch.stack(outputs, dim=0).squeeze()
        
        # batch_size * hidden_dim -> batch_size * num_user_state * emb_dim
        outputs = self.relu(self.linear(outputs))

        logits = F.log_softmax(outputs, -1)
        print("logits: ", logits.size(), last_cate.size())
        
        # Get softmax and logits
        logits = last_cate * logits
        logits = logits.sum(-1)
        
        # Calculate different metrics
        pred_last_cate = F.softmax(outputs, -1)
        print(inputs[0], pred_last_cate[0], last_cate[0])
        last_acc = (last_cate - pred_last_cate) ** 2
        last_acc = torch.sqrt(last_acc.sum(-1)).mean(0).item()
        # print("sample >>>>>>")
        # print(last_cate[0])
        # print(F.softmax(outputs, -1)[0])
        
        last_cate = last_cate.tolist()
        pred_last_cate = pred_last_cate.tolist()
        
        avg_kl_distance = 0
        for i in range(batch_size):
            avg_kl_distance += kl_divergence(last_cate[i], pred_last_cate[i])
        avg_kl_distance /= batch_size
            
        # print(label_map)
        return -logits * 0.5, -logits * 0.5, -last_acc, batch_size, avg_kl_distance
        '''