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

    def forward(self, input, label, with_graph=False, eval=False):
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
        
        avg_acc = 0
        for i in range(batch_size):
            avg_acc += (np.argmax(label[i]) == np.argmax(pred_label[i]))
        avg_acc /= batch_size
            
        # return -logits, avg_acc
        avg_acc = [0, 0 ,0]
        count = [0, 0, 0]
        for i in range(batch_size):
            avg_acc[0] += (np.argmax(label[i]) == np.argmax(pred_label[i]))
            count[0] += 1
            if pred_label[i][1] >= 0.5:
                # positive
                if label[i][1] == 1:
                    # true positive in predicted positive 
                    avg_acc[1] += 1
                count[1] += 1
            if label[i][1] == 1:
                # true positive in dataset
                if pred_label[i][1] >= 0.5:
                    avg_acc[2] += 1
                count[2] += 1

        avg_acc = [avg_acc[i] / (count[i] + 0.0001) for i in range(3)]
        # if not eval:
        #     print(avg_acc, count)

        return -logits, avg_acc[0], avg_acc[1], avg_acc[2], count

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
        loss_all, avg_acc_all, precision_all, recall_all = 0, 0, 0, 0
        count_all = [0.001, 0.001, 0.001]
        for i, batch in enumerate(train_dataloader):
            input, label = batch["input"], batch["label"]
            loss, avg_acc, precision, recall, count = self.stealthy_model(input, label)
            loss = loss.mean(0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()

            avg_acc_all += avg_acc * count[0]
            precision_all += precision * count[1]
            recall_all += recall * count[2]

            count_all[0] += count[0]
            count_all[1] += count[1]
            count_all[2] += count[2]

            if i % 10 == 0:
                self.logger.info(f"End, loss: {loss_all / (i+1)}, avg_acc: {avg_acc_all / count_all[0]}, "
                                 f"precision: {precision_all / count_all[1]}, recall: {recall_all / count_all[2]}")

        return loss_all / (i+1), avg_acc_all / (i+1)

    def eval(self, eval_dataloader):
        self.stealthy_model.eval()
        loss_all, avg_acc_all, precision_all, recall_all = 0, 0, 0, 0
        count_all = [0.001, 0.001, 0.001]
        for i, batch in enumerate(eval_dataloader):
            input, label = batch["input"], batch["label"]
            loss, avg_acc, precision, recall, count = self.stealthy_model(input, label, eval=True)
            loss = loss.mean(0)
            loss_all += loss.item()
            avg_acc_all += avg_acc * count[0]
            precision_all += precision * count[1]
            recall_all += recall * count[2]

            count_all[0] += count[0]
            count_all[1] += count[1]
            count_all[2] += count[2]

        self.logger.info(f"End, loss: {loss_all / (i+1)}, avg_acc: {avg_acc_all / count_all[0]}, "
                         f"precision: {precision_all / count_all[1]}, recall: {recall_all / count_all[2]}")

        precision = precision_all / count_all[1]
        recall = recall_all / count_all[2]
        return loss_all / (i+1), avg_acc_all / count_all[0], precision, recall, precision * recall / (precision + recall + 0.0001) * 2

def get_stealthy_dataset(base_persona, obfu_persona, batch_size=50, max_len=50):
    train_size = int(len(base_persona) * 0.7)
    val_size = int(len(base_persona) * 0.8)
    print(train_size, len(base_persona), len(obfu_persona))
    train_dataset = StealthyDataset(base_persona[:train_size], obfu_persona[:train_size], max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = StealthyDataset(base_persona[train_size:val_size], obfu_persona[train_size:val_size], max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = StealthyDataset(base_persona[val_size:], obfu_persona[val_size:], max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader