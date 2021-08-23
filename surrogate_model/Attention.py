import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, emb_dim_in, emb_dim_out):
        super(Attention, self).__init__()
        self.emb_dim_in = emb_dim_in
        self.emb_dim_out = emb_dim_out
        self.att1 = nn.Linear(self.emb_dim_in * 2, self.emb_dim_out)
        self.att2 = nn.Linear(self.emb_dim_out, self.emb_dim_out)
        self.att3 = nn.Linear(self.emb_dim_out, 1)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)
        return att
