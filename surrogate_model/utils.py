import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import YTDataset, ToTensor
torch.manual_seed(0)

def load_metadata(
    video_emb_path="../dataset/video_embeddings.hdf5",
    video_graph_path="../dataset/video_adj_list.json",
    video_id_path="../dataset/video_ids.json",
    logger=None
):
    # Load video embedding
    with h5py.File(video_emb_path, "r") as hf_emb:
        video_embeddings = hf_emb["embeddings"][:]
    emb_dim = np.shape(video_embeddings)[1]
    logger.info("video embedding dimension: {}".format(emb_dim))

    # Load video graph
    with open(video_graph_path, "r") as json_file:
        video_graph_adj_mat = json.load(json_file)
    num_videos = len(video_graph_adj_mat.keys())
    logger.info("No. of videos: {}".format(num_videos))

    # Load video id mapping
    with open(video_id_path, "r") as json_file:
        video_ids_map = json.load(json_file)
    
    return video_embeddings, video_graph_adj_mat, video_ids_map, num_videos, emb_dim


def load_dataset(
    train_data_path="../dataset/train_data.hdf5",
    test_data_path="../dataset/test_data.hdf5",
    batch_size=256,
    logger=None
):
    try:
        with h5py.File(train_data_path, "r") as train_hf:
            train_dataset = YTDataset(train_hf["input"][:], train_hf["label"][:], train_hf["label_type"][:], train_hf["mask"][:], transform=ToTensor())
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        with  h5py.File(test_data_path, "r") as test_hf:
            test_dataset = YTDataset(test_hf["input"][:], test_hf["label"][:], test_hf["label_type"][:], test_hf["mask"][:], transform=ToTensor())
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    except:
        logger.error("Failed to load training and testing dataset.")
        return None, None

def run_epoch(model, dataloader, mode="train", optimizer=None, ep=0, stat=None, logger=None, use_graph=False):
    for i, batch in enumerate(dataloader):
        # Get data
        input, label, label_type, mask = batch["input"], batch["label"], batch["label_type"], batch["mask"]
        logger.debug(input.size(), label.size(), label_type.size(), mask.size())

        if mode == "train":
            # Forward computation and back propagation
            model.train()
            optimizer.zero_grad()
            loss, acc, count, last_acc, last_count = model(input,label,label_type,mask, with_graph=use_graph)
            loss = loss.mean(0)
            loss.backward()
            optimizer.step()
        else:
            # Forward computation 
            model.eval()
            loss, acc, count, last_acc, last_count = model(input, label, label_type, mask, with_graph=use_graph)
            loss = loss.mean(0)
            
        # Print training results
        stat[f"{mode}_acc"] += acc * count
        stat[f"{mode}_count"] += count
        stat[f"{mode}_last_acc"] += last_acc * last_count
        stat[f"{mode}_last_count"] += last_count
        stat[f"{mode}_loss"] += loss.item() * count
        logger.info("{}ing, ep: {}, step: {}, loss: {}, acc: {}, home_acc: {}.".format(mode, ep, i, loss.item(), acc, last_acc))
    
    # Print final training results
    stat[f"{mode}_acc"] /= stat[f"{mode}_count"]
    stat[f"{mode}_last_acc"] /= stat[f"{mode}_last_count"]
    stat[f"{mode}_loss"] /= stat[f"{mode}_count"]
    logger.info("End {}ing, ep: {}, loss: {}, acc: {}, home_acc: {}.".format(mode, ep, stat[f"{mode}_loss"], stat[f"{mode}_acc"], stat[f"{mode}_last_acc"]))
    return stat