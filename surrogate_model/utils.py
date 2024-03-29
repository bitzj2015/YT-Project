import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import YTDataset, ToTensor
from tqdm import tqdm
import time
torch.manual_seed(0)


def load_metadata(
    video_emb_path="../dataset/video_embeddings.hdf5",
    video_graph_path="../dataset/video_adj_list.json",
    video_id_path="../dataset/video_ids.json",
    logger=None
):
    # Load video embedding
    with h5py.File(video_emb_path, "r") as hf_emb:
        video_embeddings = hf_emb["embeddings"][:].astype("float32")
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
            train_dataset = YTDataset(
                train_hf["input"][:], train_hf["label"][:], 
                train_hf["label_type"][:], train_hf["mask"][:], 
                train_hf["last_label"][:], train_hf["last_label_p"][:], train_hf["last_label_type"][:], 
                train_hf["last_cate_norm"][:],
                transform=ToTensor()
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        with  h5py.File(test_data_path, "r") as test_hf:
            dataset = YTDataset(
                test_hf["input"][:], test_hf["label"][:], 
                test_hf["label_type"][:], test_hf["mask"][:], 
                test_hf["last_label"][:], test_hf["last_label_p"][:], test_hf["last_label_type"][:], 
                test_hf["last_cate_norm"][:],
                transform=ToTensor()
            )
            dataset_size = len(dataset)
            print(dataset_size)
            val_dataset_size = int(0.5 * dataset_size)
            test_dataset_size = dataset_size - val_dataset_size
            test_dataset, val_dataset = torch.utils.data.random_split(dataset, [test_dataset_size, val_dataset_size], generator=torch.Generator().manual_seed(0))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    except:
        logger.error("Failed to load training and testing dataset.")
        return None, None


def run_regression_epoch(model, dataloader, mode="train", optimizer=None, ep=0, stat=None, logger=None, use_graph=False):
    for i, batch in tqdm(enumerate(dataloader)):
        # Get data
        input, label, label_type, mask = batch["input"], batch["label"], batch["label_type"], batch["mask"]
        last_label = batch["last_label"]
        last_label_type = batch["last_label_type"]
        last_label_p = batch["last_label_p"]
        logger.debug(input.size(), label.size(), label_type.size(), mask.size(), last_label.size(), last_label_type.size())

        if mode == "train":
            # Forward computation and back propagation
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            loss_pos, loss_neg, last_acc, last_count, last_acc_ch = model(input, last_label, last_label_p, last_label_type, mask, with_graph=use_graph)
            print("forward:", time.time()-start_time)
            loss_pos = loss_pos.mean(0)
            loss_neg = loss_neg.mean(0)
            # print(loss)
            (loss_pos + loss_neg).backward()
            optimizer.step()
            print("backward:", time.time()-start_time)
 
        else:
            # Forward computation 
            model.eval()
            loss_pos, loss_neg, last_acc, last_count, last_acc_ch = model(input, last_label, last_label_p, last_label_type, mask, with_graph=use_graph)
            loss_pos = loss_pos.mean(0)
            loss_neg = loss_neg.mean(0)
            
        # Print training results
        stat[f"{mode}_last_acc"] += last_acc * last_count
        stat[f"{mode}_last_acc_ch"] += last_acc_ch * last_count
        stat[f"{mode}_last_count"] += last_count
        stat[f"{mode}_loss_pos"] += loss_pos.item() * last_count
        stat[f"{mode}_loss_neg"] += loss_neg.item() * last_count
        logger.info("{}ing, ep: {}, step: {}, loss_pos: {}, loss_neg: {}, home_acc: {}, ch_acc: {}.".format(mode, ep, i, loss_pos.item(), loss_neg.item(), last_acc, last_acc_ch))
    
    # Print final training results
    stat[f"{mode}_last_acc"] /= stat[f"{mode}_last_count"]
    stat[f"{mode}_last_acc_ch"] /= stat[f"{mode}_last_count"]
    stat[f"{mode}_loss_pos"] /= stat[f"{mode}_last_count"]
    stat[f"{mode}_loss_neg"] /= stat[f"{mode}_last_count"]
    logger.info("End {}ing, ep: {}, loss_pos: {}, loss_neg: {}, home_acc: {}, ch_acc: {}.".format(mode, ep, stat[f"{mode}_loss_pos"], stat[f"{mode}_loss_neg"], stat[f"{mode}_last_acc"], stat[f"{mode}_last_acc_ch"]))
    return stat