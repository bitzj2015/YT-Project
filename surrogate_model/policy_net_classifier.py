import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

class PolicyNetClassifier(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, num_videos=127085):
        super(PolicyNetClassifier,self).__init__()

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, num_videos)
        self.video_embeddings = video_embeddings # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.relu = nn.ReLU()
        self.train()

    def forward(self, inputs, label, label_type, mask, with_graph=False):
        # inputs = torch.reshape(inputs, (-1,))
        # inputs = self.video_embeddings(inputs.tolist())
        # inputs = torch.reshape(inputs, (batch_size, seq_len, self.emb_dim))
        # print(inputs.size())
        batch_size, seq_len = inputs.shape
        inputs = inputs * mask
        inputs = self.video_embeddings(inputs.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        mask = mask.unsqueeze(2)
        

        # batch_size * seq_len * emb_dim -> batch_size * seq_len * (2*hidden_dim)
        outputs, _ = self.lstm(inputs)

        # batch_size * seq_len * (2*hidden_dim) -> batch_size * seq_len * num_videos
        outputs = self.relu(self.linear(outputs))

        # batch_size * seq_len * num_videos
        label = label.long()
        # print(outputs.size(), label.size())
        logits = torch.gather(F.log_softmax(outputs, -1), -1, label)


        # for i in range(len(mask)):
        #     for j in range(sum(mask[i])):
        #         label_type[i][j] *=0
        #         label_type[i][j][0] = 1
        # logits = torch.gather(F.log_softmax(outputs, -1), -1, label)
        
        # Get softmax and logits
        logits = mask * label_type * logits
        logits = logits.mean(2) # / (label_type.sum(2) + 1e-10)
        logits = logits.mean(1)

        _, rec_outputs = torch.topk(F.softmax(outputs, -1), k=100, dim=-1)
        rec_outputs = rec_outputs.tolist()
        label = label.tolist()
        label_type = label_type.tolist()
        mask = mask.squeeze(2).tolist()
        count = 0
        acc = 0
        last_acc = 0
        last_count = 0
        for i in range(len(mask)):
            for j in range(sum(mask[i])):
                num_rec = sum(label_type[i][j])
                label_map = dict(zip(label[i][j][:num_rec], [0 for _ in range(num_rec)]))
                for k in range(num_rec):
                    if rec_outputs[i][j][k] in label_map.keys():
                        acc += 1
                        if j == sum(mask[i]) - 1:
                            last_acc += 1
                            # print(rec_outputs[i][j][k], inputs[i][j])
                count += num_rec
                if j == sum(mask[i]) - 1:
                    last_count += num_rec
                    # print(label[i][j][:num_rec],rec_outputs[i][j][:num_rec])
            # break

        return -logits, acc/count, count, last_acc/last_count, last_count