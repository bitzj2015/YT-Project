import torch
import numpy as np
from torch.utils.data import Dataset

class YTDataset(Dataset):
    def __init__(
        self,
        input_data,
        label_data,
        label_type_data,
        mask_data,
        transform=None
    ):
        self.input = input_data
        self.label = label_data
        self.label_type = label_type_data
        self.mask_data = mask_data
        self.transform = transform

    def __len__(self):
        return np.shape(self.label)[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        input_data = self.input[idx]
        label_data = self.label[idx]
        label_type_data = self.label_type[idx]
        mask_data = self.mask_data[idx]
        
        sample = {"input":input_data, "label":label_data, "label_type": label_type_data, "mask": mask_data}
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self, sample):
        input, label, label_type, mask = sample["input"], sample["label"], sample["label_type"], sample["mask"]
        return {"input":torch.from_numpy(input), "label":torch.from_numpy(label), "label_type": torch.from_numpy(label_type), "mask": torch.from_numpy(mask)}