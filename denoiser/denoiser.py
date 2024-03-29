import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
random.seed(1024)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def kl_divergence(p, q):
	return sum(p[i] * np.log2((p[i]+1e-9)/(q[i]+1e-9)) for i in range(len(p)))

class DenoiserNet(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, device="cpu", base=False):
        super(DenoiserNet,self).__init__()
        # gpu/cpu
        self.device = "cpu"

        # Used to encode non-obfuscated videos
        self.lstm_vu = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim, #256
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        ).to(self.device)

        # Used to encode obfuscated videos
        self.lstm_vo = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=False,
            dropout=0.2,
            batch_first=True
        ).to(self.device)

        # Used to encode obfuscated recommended video distribution
        self.linear_ro = nn.Linear(154, hidden_dim).to(self.device)

        # Used to predict non-obfuscated recommended video distribution
        self.linear_ru = nn.Linear(hidden_dim * 3, 154).to(self.device)

        # Others
        # self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.base = base
        self.train()

    def forward(self, input_vu, input_vo, label_ro, label_ru, with_graph=False):
        # Get embeddings for non-obfuscated videos
        batch_size, seq_len = input_vu.shape
        input_vu = self.video_embeddings[input_vu.reshape(-1).tolist()].reshape(batch_size, seq_len, self.emb_dim)

        # Get embeddings for obfuscated videos
        batch_size, seq_len = input_vo.shape
        input_vo = self.video_embeddings[input_vo.reshape(-1).tolist()].reshape(batch_size, seq_len, self.emb_dim)

        label_ro = label_ro.to(self.device)
        label_ru = label_ru.to(self.device)

        # Encode non-obfuscated videos
        encoded_vu, _ = self.lstm_vu(input_vu)

        # Encode obfuscated videos
        encoded_vo, _ = self.lstm_vo(input_vo)

        # Encode obfuscated recommended video distribution
        encoded_ro = self.relu(self.linear_ro(label_ro))

        # Predict non-obfuscated recommended video distribution
        if self.base:
            decoded_ru = self.linear_ru(torch.cat([encoded_vu[:, -1] * 1, encoded_vo[:, -1] * 0, encoded_ro * 0], axis=-1))
        else:
            decoded_ru = self.linear_ru(torch.cat([encoded_vu[:, -1], encoded_vo[:, -1], encoded_ro], axis=-1))

        pred_label_ru = F.softmax(decoded_ru, -1)
        # pred_label_ru = pred_label_ru * (pred_label_ru > 1e-4)
        logits = torch.log(pred_label_ru)
        # print("logits: ", logits.size(), label_ru.size())
        
        # Get softmax and logits
        # logits = label_ru * logits
        logits = pred_label_ru * (logits - torch.log(label_ru + 1e-9))
        logits = logits.sum(-1)
        
        label_ru = label_ru.tolist()
        pred_label_ru = pred_label_ru.tolist()
        
        avg_kl_distance = 0
        for i in range(batch_size):
            avg_kl_distance += kl_divergence(pred_label_ru[i], label_ru[i])
        avg_kl_distance /= batch_size
            
        return logits, avg_kl_distance
        # outputs = torch.sigmoid(decoded_ru)
        # loss = label_ru * torch.log(outputs) + (1 - label_ru) * torch.log(1 - outputs)
        # pred = (outputs >= 0.5).int()
        # print((pred == label_ru).sum() / pred.size(0) / pred.size(1))

        # loss = -loss.mean()
        # hamming_dis = torch.abs(pred - label_ru).sum(-1)

        # return loss, hamming_dis.mean()


class DenoiserDataset(Dataset):
    def __init__(self, base_persona, obfu_persona, base_rec, obfu_rec, max_len=50):
        self.base_persona = base_persona
        self.obfu_persona = obfu_persona
        self.base_rec = base_rec
        self.obfu_rec = obfu_rec
        self.max_len = max_len

    def __len__(self):
        return np.shape(self.base_persona)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        base_persona = self.base_persona[idx] + [0 for _ in range(40 - len(self.base_persona[idx]))]
        obfu_persona = self.obfu_persona[idx] + [0 for _ in range(self.max_len - len(self.obfu_persona[idx]))]
        base_rec = self.base_rec[idx]
        obfu_rec = self.obfu_rec[idx]

        sample = {
            "input_vu": torch.from_numpy(np.array(base_persona).astype("int32")), 
            "input_vo": torch.from_numpy(np.array(obfu_persona).astype("int32")), 
            "label_ru": torch.from_numpy(np.array(base_rec).astype("float32")), 
            "label_ro": torch.from_numpy(np.array(obfu_rec).astype("float32"))
        }
        return sample


class Denoiser(object):
    def __init__(self, denoiser_model, optimizer, logger):
        self.denoiser_model = denoiser_model
        self.optimizer = optimizer
        self.logger = logger

    def train(self, train_dataloader):
        self.denoiser_model.train()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(train_dataloader):
            input_vu, input_vo, label_ro, label_ru = batch["input_vu"], batch["input_vo"], batch["label_ro"], batch["label_ru"]
            loss, kl_div = self.denoiser_model(input_vu, input_vo, label_ro, label_ru)
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
        self.denoiser_model.eval()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(eval_dataloader):
            input_vu, input_vo, label_ro, label_ru = batch["input_vu"], batch["input_vo"], batch["label_ro"], batch["label_ru"]
            loss, kl_div = self.denoiser_model(input_vu, input_vo, label_ro, label_ru)
            loss = loss.mean(0)
            loss_all += loss.item()
            kl_div_all += kl_div
        self.logger.info(f"End, loss: {loss_all / (i+1)}, kl_div: {kl_div_all / (i+1)}")

        return loss_all / (i+1), kl_div_all / (i+1)

def get_denoiser_dataset(base_persona, obfu_persona, base_rec, obfu_rec, batch_size=50, max_len=50, all_eval=False):
    c = list(zip(base_persona, obfu_persona, base_rec, obfu_rec))
    random.shuffle(c)
    base_persona, obfu_persona, base_rec, obfu_rec = zip(*c)

    if all_eval:
        train_dataset = DenoiserDataset(base_persona, obfu_persona, base_rec, obfu_rec, max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=False )
        return train_loader
    else:
        train_size = int(len(base_persona) * 0.7)
        print(train_size, len(base_persona), len(obfu_persona), len(base_rec), len(obfu_rec))
        train_dataset = DenoiserDataset(base_persona[:train_size], obfu_persona[:train_size], base_rec[:train_size], obfu_rec[:train_size], max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)

        val_size = int(len(base_persona) * 0.8)
        val_dataset = DenoiserDataset(base_persona[train_size:val_size], obfu_persona[train_size:val_size], base_rec[train_size:val_size], obfu_rec[train_size:val_size], max_len)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = DenoiserDataset(base_persona[val_size:], obfu_persona[val_size:], base_rec[val_size:], obfu_rec[val_size:], max_len)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

        return train_loader, val_loader, test_loader
            