import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim, video_embeddings, batch_size=1024):
        super(PolicyNet,self).__init__()

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        self.linear = nn.Linear(2*hidden_dim, embed_dim)
        self.video_embeddings = video_embeddings # num_videos * emb_dim
        self.hx = torch.zeros(4, batch_size, hidden_dim)
        self.cx = torch.zeros(4, batch_size, hidden_dim)
        self.embed_dim = embed_dim
        self.train()

    def forward(self, inputs, label, label_type, mask):
        # inputs = torch.reshape(inputs, (-1,))
        # inputs = self.video_embeddings(inputs.tolist())
        # inputs = torch.reshape(inputs, (batch_size, seq_len, self.embed_dim))
        # print(inputs.size())

        batch_size, seq_len = inputs.shape
        inputs = self.video_embeddings(inputs.reshape(-1).tolist()).reshape(batch_size, seq_len, self.embed_dim)
        mask = mask.unsqueeze(2)
        inputs = inputs * mask

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * (2*hidden_dim)
        outputs, _ = self.lstm(inputs)

        # batch_size * seq_len * (2*hidden_dim) -> batch_size * seq_len * 1 * emb_dim
        outputs = torch.tanh(self.linear(outputs)).unsqueeze(2)

        # batch_size * seq_len * emb_dim * num_videos
        label = label.long()
        batch_size, seq_len, label_len = label.shape
        sample_embedding = self.video_embeddings(label.reshape(-1).tolist()).reshape(batch_size, seq_len, label_len, self.embed_dim)
        sample_embedding = sample_embedding.permute(0,1,3,2)

        # label = torch.reshape(label.long(), (-1,))
        # sample_embedding = self.video_embeddings(label.tolist())
        # sample_embedding = torch.reshape(sample_embedding, (batch_size, seq_len, 100, self.embed_dim))
        # sample_embedding = sample_embedding.permute(0,1,3,2)

        # batch_size * seq_len * 1 * emb_dim -> batch_size * seq_len * emb_dim * num_videos
        logits = torch.matmul(outputs, sample_embedding) / ((torch.norm(outputs, dim=3).unsqueeze(3) * torch.norm(sample_embedding, dim=2).unsqueeze(2)) + 1e-10)
        logits = logits.squeeze(2)

        # Get softmax and logits
        logits_ = mask * (1-label_type) * logits
        logits_ = logits_.sum(2) / ((1-label_type).sum(2) + 1e-10)
        print("logits_", logits_[0:10,-1], (1-label_type).sum(2)[0:10,-1])
        logits_ = logits_.mean(1)

        logits = mask * label_type * (1-logits)
        logits = logits.sum(2) / (label_type.sum(2) + 1e-10)
        print("logits", 1-logits[0:10,-1], label_type.sum(2)[0:10,-1])
        logits = logits.mean(1)
        

        return logits, logits_, outputs