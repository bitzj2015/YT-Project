import torch
import torch.nn as nn
from Attention import Attention
from tqdm import tqdm
import numpy as np

class Social_Aggregator(nn.Module):
    """
    Graph Aggregator: for aggregating embeddings of video neighbors.
    """

    def __init__(self, video_embeddings, emb_dim, add_edge=False, device="cpu"):
        super(Social_Aggregator, self).__init__()

        self.device = device
        self.video_embeddings = video_embeddings
        self.emb_dim = emb_dim
        self.add_edge = add_edge
        # if self.add_edge:
        #     self.att = Attention(self.emb_dim + 1, self.emb_dim)
        # else:
        self.att = Attention(self.emb_dim, self.emb_dim)
        # self.att_neigh = nn.Linear(self.emb_dim, 1)
        # self.att_node = nn.Linear(self.emb_dim, 1)

    def forward(self, video_nodes, video_neighs_list, video_neighs_weights_list):

        # Define the neighborhood embedding matrix
        embed_matrix = torch.empty(len(video_nodes), self.emb_dim, dtype=torch.float).to(self.device)
        # att_neigh_embeddings = self.att_neigh(self.video_embeddings)
        # att_node_embeddings = self.att_neigh(self.video_embeddings)

        # Get the neighborhood embedding of each video node
        for i in tqdm(range(len(video_nodes))):

            # Get the neighborhood video node list of each video
            video_neighs = [int(idx) for idx in video_neighs_list[i]]
            # att_w = torch.Tensor([int(idx) for idx in video_neighs_weights_list[i]]).unsqueeze(1)

            # Get the number of neighborhood video nodes
            num_video_neighs = len(video_neighs)

            # Get the embeddings of neighborhood video nodes
            # if self.add_edge:
            #     video_neighs_weights = np.array([int(idx) for idx in video_neighs_weights_list[i]]).astype("float32")
            #     neigh_video_emb = torch.cat([self.video_embeddings[list(video_neighs)], torch.from_numpy(video_neighs_weights/100).unsqueeze(1)],axis=1)
            # else:
            neigh_video_emb = self.video_embeddings[list(video_neighs)]
            # neigh_video_emb = att_neigh_embeddings[list(video_neighs)]

            # Get the embedding of current video node
            # if self.add_edge:
            #     video_emb = torch.cat([self.video_embeddings[video_nodes[i]], torch.from_numpy(np.array([0]))], axis=0)
            # else:
            video_emb = self.video_embeddings[video_nodes[i]]
            # video_emb = att_node_embeddings[video_nodes[i]].unsqueeze(0)
            # att_w = torch.tanh(neigh_video_emb+video_emb)

            # Use attention layer to get the neighborhood embedding
            att_w = self.att(neigh_video_emb, video_emb, num_video_neighs)
            att_history = torch.matmul(self.video_embeddings[list(video_neighs)].t(), att_w).t()

            # Store the neighborhood embedding
            embed_matrix[i] = att_history.squeeze(0)

        return embed_matrix
