import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

class StealthyNet(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, device="cpu", base=False):
        super(StealthyNet,self).__init__()
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

    def forward(self, input, label, with_graph=False):
        # Get embeddings for non-obfuscated videos
        batch_size, seq_len = input.shape
        input = self.video_embeddings[input.reshape(-1).tolist()].reshape(batch_size, seq_len, self.emb_dim)

        # Encode non-obfuscated videos
        encoded, _ = self.lstm(input)


        # Predict non-obfuscated recommended video distribution
        decoded = self.linear(encoded[:, -1])

        logits = F.log_softmax(decoded, -1)
        # print("logits: ", logits.size(), label_ru.size())
        
        # Get softmax and logits
        logits = label * logits
        logits = logits.sum(-1)
        
        # Calculate different metrics
        pred_label = F.softmax(decoded, -1)
        
        label = label.tolist()
        pred_label = pred_label.tolist()
        
        avg_kl_distance = 0
        for i in range(batch_size):
            avg_kl_distance += (np.argmax(label[i]) == np.argmax(pred_label[i]))
        avg_kl_distance /= batch_size
            
        return -logits, avg_kl_distance


class StealthyDataset(Dataset):
    def __init__(self, base_persona, obfu_persona, max_len=50):
        self.persona = base_persona + obfu_persona
        self.label = [[1,0] for _ in range(len(base_persona))] + [[0,1] for _ in range(len(obfu_persona))]
        self.max_len = max_len

    def __len__(self):
        return np.shape(self.persona)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()

        sample = {
            "input": torch.from_numpy(np.array(self.persona[idx]).astype("int32")), 
            "label": torch.from_numpy(np.array(self.label[idx]).astype("float32"))
        }
        return sample


class Stealthy(object):
    def __init__(self, stealthy_model, optimizer, logger):
        self.stealthy_model = stealthy_model
        self.optimizer = optimizer
        self.logger = logger

    def train(self, train_dataloader):
        self.stealthy_model.train()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(train_dataloader):
            input, label = batch["input"], batch["label"]
            loss, kl_div = self.stealthy_model(input, label)
            loss = loss.mean(0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()
            kl_div_all += kl_div
            if i % 10 == 0:
                self.logger.info(f"Step: {i}, loss: {loss_all / (i+1)}, kl_div: {kl_div_all / (i+1)}")

        return loss_all / (i+1), kl_div_all / (i+1)

    def eval(self, eval_dataloader):
        self.stealthy_model.eval()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(eval_dataloader):
            input, label = batch["input"], batch["label"]
            loss, kl_div = self.stealthy_model(input, label)
            loss = loss.mean(0)
            loss_all += loss.item()
            kl_div_all += kl_div
        self.logger.info(f"End, loss: {loss_all / (i+1)}, kl_div: {kl_div_all / (i+1)}")

        return loss_all / (i+1), kl_div_all / (i+1)

def get_stealthy_dataset(base_persona, obfu_persona, batch_size=50, max_len=50):
    train_size = int(len(base_persona) * 0.8)
    print(train_size, len(base_persona), len(obfu_persona))
    train_dataset = StealthyDataset(base_persona[:train_size], obfu_persona[:train_size], max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = StealthyDataset(base_persona[train_size:], obfu_persona[train_size:], max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
            