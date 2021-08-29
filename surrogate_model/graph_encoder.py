import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphEncoder(nn.Module):

    def __init__(self, video_embeddings, emb_dim, video_graph_adj_mat, aggregator, device="cpu"):
        super(GraphEncoder, self).__init__()

        self.video_embeddings = video_embeddings
        self.video_graph_adj_mat = video_graph_adj_mat
        self.aggregator = aggregator
        self.emb_dim = emb_dim
        self.device = device
        self.linear1 = nn.Linear(2 * self.emb_dim, self.emb_dim)

    def forward(self, video_nodes, with_graph=True):

        if with_graph == True:
            # Get the neighborhood nodes of each video node
            video_neighs_list = []
            video_neighs_weights_list = []
            for video_node in video_nodes:
                video_neighs_list.append(list(self.video_graph_adj_mat[str(video_node)].keys()))
                video_neighs_weights_list.append(list(self.video_graph_adj_mat[str(video_node)].values()))

            # Get the neighborhood aggregation embedding of each video node
            neigh_feats = self.aggregator.forward(video_nodes, video_neighs_list, video_neighs_weights_list)

            # Get the embedding of each video node
            self_feats = self.video_embeddings[video_nodes].to(self.device)
            
            # Combine the embedding and neighborhood embedding of each video node
            combined = torch.cat([self_feats, neigh_feats], dim=1)

            # Fully-connected neural network
            combined = F.relu(self.linear1(combined))
        else:
            combined = self.video_embeddings[video_nodes]

        return combined
