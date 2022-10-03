from denoiser import *
import torch
import pandas as pd
import h5py
import torch.optim as optim
import logging
import numpy as np
import tensorflow as tf
from memory_profiler import profile
import time

@profile
# ##

def calc():
    logging.basicConfig(
        filename=f"../../param/log.txt",
        filemode='w',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.INFO
    )
    logger=logging.getLogger() 
    logger.setLevel(logging.INFO) 

    with h5py.File(f"../../param/video_embeddings_40_June_aug.hdf5", "r") as hf_emb:
        video_embeddings = hf_emb["embeddings"][:].astype("float32")
    video_embeddings = torch.from_numpy(video_embeddings)


    denoiser_model = DenoiserNet(emb_dim=404, hidden_dim=256, video_embeddings=video_embeddings)
    denoiser_optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)
    denoiser = Denoiser(denoiser_model=denoiser_model, optimizer=denoiser_optimizer, logger=logger)
    denoiser.denoiser_model = torch.load(f"../../param/denoiser_0.2_rl.pkl")

    I=np.random.randint(100,size=(1,40))
    Iprime=np.random.randint(100,size=(1,40))
    #Oprime=np.random.randint(100,size=(1,154))
    Oprime=np.random.rand(1,154)
    tensor1 = torch.from_numpy(Oprime).type(torch.FloatTensor)
    start_time = time.time()
    denoiser.denoiser_model.get_rec(I,Iprime,tensor1)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    
    calc()
    
    
