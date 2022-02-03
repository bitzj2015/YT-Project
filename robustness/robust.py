import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def avg_accergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

class RobustNet(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, device="cpu", base=False):
        super(RobustNet,self).__init__()
        # gpu/cpu
        self.device = device

        # Used to encode non-obfuscated videos
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim, #256
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        ).to(self.device)

        # Used to predict non-obfuscated recommended video distribution
        self.linear = nn.Linear(hidden_dim, 2).to(self.device)

        # Others
        # self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.base = base
        self.train()

    def forward(self, input, label, mask, with_graph=False):
        # Get embeddings for non-obfuscated videos
        batch_size, seq_len = input.shape
        input = self.video_embeddings[input.reshape(-1).tolist()].reshape(batch_size, seq_len, self.emb_dim)

        # Encode non-obfuscated videos
        encoded, _ = self.lstm(input)


        # Predict non-obfuscated recommended video distribution
        decoded = self.linear(encoded)

        logits = F.log_softmax(decoded, -1)
        # print("logits: ", logits.size(), label_ru.size())
        
        # Get softmax and logits
        # print(logits.size(), label.size(), mask.unsqueeze(-1).size())
        logits = label * logits * mask.unsqueeze(-1)
        logits = logits.sum(-1).sum(-1) / mask.sum(-1)
        
        # Calculate different metrics
        pred_label = F.softmax(decoded, -1)
        
        label = label.tolist()
        pred_label = pred_label.tolist()
        
        avg_acc = [0, 0 ,0]
        count = [0, 0, 0]
        for i in range(batch_size):
            for j in range(len(mask[i])):
                avg_acc[0] += (np.argmax(label[i][j]) == np.argmax(pred_label[i][j]))
                count[0] += 1
                if label[i][j][0] == 1:
                    avg_acc[1] += (np.argmax(label[i][j]) == np.argmax(pred_label[i][j]))
                    count[1] += 1
                else:
                    avg_acc[2] += (np.argmax(label[i][j]) == np.argmax(pred_label[i][j]))
                    count[2] += 1
        avg_acc = [avg_acc[i] / count[i] for i in range(3)]
            
        return -logits, avg_acc[0], avg_acc[1], avg_acc[2]


class RobustDataset(Dataset):
    def __init__(self, obfu_persona, obfu_rec, max_len=50):
        self.persona = obfu_persona
        self.label = obfu_rec
        self.max_len = max_len

    def __len__(self):
        return np.shape(self.persona)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()

        sample = {
            "input": torch.from_numpy(np.array(
                self.persona[idx] + [0 for _ in range(self.max_len-len(self.persona[idx]))]
                ).astype("float32")), 
            "label": torch.from_numpy(np.array(
                self.label[idx] + [[0, 1] for _ in range(self.max_len-len(self.persona[idx]))]
                ).astype("int32")),
            "mask": torch.from_numpy(np.array(
                [1 for _ in range(len(self.persona[idx]))] + [0 for _ in range(self.max_len-len(self.persona[idx]))]
                ).astype("int32"))
        }
        return sample


class Robust(object):
    def __init__(self, robust_model, optimizer, logger):
        self.robust_model = robust_model
        self.optimizer = optimizer
        self.logger = logger

    def train(self, train_dataloader):
        self.robust_model.train()
        loss_all, avg_acc_all, avg_acc_0_all, avg_acc_1_all = 0, 0, 0, 0
        for i, batch in enumerate(train_dataloader):
            input, label, mask = batch["input"], batch["label"], batch["mask"]
            loss, avg_acc, avg_acc_0, avg_acc_1 = self.robust_model(input, label, mask)
            loss = loss.mean(0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()
            avg_acc_all += avg_acc
            avg_acc_0_all += avg_acc_0
            avg_acc_1_all += avg_acc_1
            if i % 10 == 0:
                self.logger.info(f"End, loss: {loss_all / (i+1)}, avg_acc: {avg_acc_all / (i+1)}, "
                                 f"avg_acc_0: {avg_acc_0_all / (i+1)}, avg_acc_1: {avg_acc_1_all / (i+1)}")

        return loss_all / (i+1), avg_acc_all / (i+1)

    def eval(self, eval_dataloader):
        self.robust_model.eval()
        loss_all, avg_acc_all, avg_acc_0_all, avg_acc_1_all = 0, 0, 0, 0
        for i, batch in enumerate(eval_dataloader):
            input, label, mask = batch["input"], batch["label"], batch["mask"]
            loss, avg_acc, avg_acc_0, avg_acc_1 = self.robust_model(input, label, mask)
            loss = loss.mean(0)
            loss_all += loss.item()
            avg_acc_all += avg_acc
            avg_acc_0_all += avg_acc_0
            avg_acc_1_all += avg_acc_1
        self.logger.info(f"End, loss: {loss_all / (i+1)}, avg_acc: {avg_acc_all / (i+1)}, "
                         f"avg_acc_0: {avg_acc_0_all / (i+1)}, avg_acc_1: {avg_acc_1_all / (i+1)}")

        return loss_all / (i+1), avg_acc_all / (i+1)

def get_robust_dataset(obfu_persona, obfu_rec, batch_size=50, max_len=50):
    train_size = int(len(obfu_persona) * 0.8)
    print(train_size, len(obfu_persona), len(obfu_persona))
    train_dataset = RobustDataset(obfu_persona[:train_size], obfu_rec[:train_size], max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = RobustDataset(obfu_persona[train_size:], obfu_rec[train_size:], max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader