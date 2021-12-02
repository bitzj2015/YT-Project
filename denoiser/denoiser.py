import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

def kl_divergence(p, q):
	return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))

class DenoiserNet(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim, video_embeddings, device="cpu"):
        super(DenoiserNet,self).__init__()
        # gpu/cpu
        self.device = device

        # Used to encode non-obfuscated videos
        self.lstm_vu = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
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
        self.linear_ro = nn.Linear(18, hidden_dim).to(self.device)

        # Used to predict non-obfuscated recommended video distribution
        self.linear_ru = nn.Linear(hidden_dim * 3, 18).to(self.device)

        # Others
        # self.graph_embeddings = graph_embeddings
        self.video_embeddings = video_embeddings.to(self.device) # num_videos * emb_dim
        self.emb_dim = emb_dim
        self.tanh = nn.Tanh().to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.train()

    def infer(self, input_vu, input_vo, label_ro, with_graph=False):
        self.eval()
        batch_size, seq_len = input_vu.shape
        # input_vu = self.graph_embeddings(input_vu.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        input_vu = self.video_embeddings[input_vu.reshape(-1).tolist()].reshape(batch_size, 1, seq_len, self.emb_dim)

        batch_size, seq_len = input_vo.shape
        # input_vo = self.graph_embeddings(input_vo.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)
        input_vo = self.video_embeddings[input_vo.reshape(-1).tolist()].reshape(batch_size, 1, seq_len, self.emb_dim)

        label_ro = label_ro.to(self.device)

        encoded_vu, _ = self.lstm_vu(input_vu)
        encoded_vo, _ = self.lstm_vu(input_vo)
        encoded_ro = self.relu(self.linear_ro(label_ro))
        decoded_ru = self.relu(self.linear_ru(torch.stack([encoded_vu, encoded_vo, encoded_ro], axis=-1)))
        outputs = F.softmax(decoded_ru, -1)
         
        return outputs.detach().cpu().numpy()

    def forward(self, input_vu, input_vo, label_ro, label_ru, with_graph=False):
        # Get embeddings for non-obfuscated videos
        batch_size, seq_len = input_vu.shape
        input_vu = self.graph_embeddings(input_vu.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)

        # Get embeddings for obfuscated videos
        batch_size, seq_len = input_vo.shape
        input_vo = self.graph_embeddings(input_vo.reshape(-1).tolist(), with_graph).reshape(batch_size, seq_len, self.emb_dim)

        label_ro = label_ro.to(self.device)
        label_ru = label_ru.to(self.device)

        # Encode non-obfuscated videos
        encoded_vu, _ = self.lstm_vu(input_vu)

        # Encode obfuscated videos
        encoded_vo, _ = self.lstm_vu(input_vo)

        # Encode obfuscated recommended video distribution
        encoded_ro = self.relu(self.linear_ro(label_ro))

        # Predict non-obfuscated recommended video distribution
        decoded_ru = self.relu(self.linear_ru(torch.stack([encoded_vu, encoded_vo, encoded_ro], axis=-1)))

        logits = F.log_softmax(decoded_ru, -1)
        print("logits: ", logits.size(), label_ru.size())
        
        # Get softmax and logits
        logits = label_ru * logits
        logits = logits.sum(-1)
        
        # Calculate different metrics
        pred_label_ru = F.softmax(decoded_ru, -1)
        
        label_ru = label_ru.tolist()
        pred_label_ru = pred_label_ru.tolist()
        
        avg_kl_distance = 0
        for i in range(batch_size):
            avg_kl_distance += kl_divergence(label_ru[i], pred_label_ru[i])
        avg_kl_distance /= batch_size
            
        return -logits, avg_kl_distance


class DenoiserDataset(Dataset):
    def __init__(self, base_persona, obfu_persona, base_rec, obfu_rec):
        self.base_persona = base_persona
        self.obfu_persoan = obfu_persona
        self.base_rec = base_rec
        self.obfu_rec = obfu_rec

    def __len__(self):
        return np.shape(self.base_persona)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        base_persona = self.base_persona[idx]
        obfu_persona = self.obfu_persona[idx]
        base_rec = self.base_rec[idx]
        obfu_rec = self.obfu_rec[idx]

        sample = {"input_vu": base_persona, "input_vo": obfu_persona, "label_ru": base_rec, "label_ro": obfu_rec}
        return sample


class Denoiser(object):
    def __init__(self, denoiser_model, optimizer):
        self.denoiser_model = denoiser_model
        self.optimizer = optimizer

    def train(self, train_dataloader):
        self.denoiser_model.train()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(train_dataloader):
            input_vu, input_vo, label_ro, label_ru = batch["input_vu"], batch["input_vo"], batch["label_ro"], batch["label_ru"]
            loss, kl_div = self.denoiser_model(input_vu, input_vo, label_ro, label_ru)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()
            kl_div_all += kl_div

        return loss_all / (i+1), kl_div / (i+1)

    def eval(self, eval_dataloader):
        self.denoiser_model.eval()
        loss_all, kl_div_all = 0, 0
        for i, batch in enumerate(eval_dataloader):
            input_vu, input_vo, label_ro, label_ru = batch["input_vu"], batch["input_vo"], batch["label_ro"], batch["label_ru"]
            loss, kl_div = self.denoiser_model(input_vu, input_vo, label_ro, label_ru)
            loss_all += loss.item()
            kl_div_all += kl_div

        return loss_all / (i+1), kl_div / (i+1)
            