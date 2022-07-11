import torch
from graph_encoder import GraphEncoder
from graph_aggregator import GraphAggregator
from policy_net_regression_cate import PolicyNetRegression
from utils_cate import *
from constants import *
import torch.optim as optim
import logging
import argparse
torch.manual_seed(0)


parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--video-emb', dest="video_emb_path", type=str, default=f"{ROOT_PATH}/dataset/video_embeddings_{VERSION}_aug.hdf5")
parser.add_argument('--video-graph', dest="video_graph_path", type=str, default=f"../dataset/video_adj_list_final_w.json")
parser.add_argument('--video-id', dest="video_id_path", type=str, default=f"{ROOT_PATH}/dataset/video_ids_{VERSION}.json")
parser.add_argument('--train-data', dest="train_data_path", type=str, default=f"{ROOT_PATH}/dataset/train_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--test-data', dest="test_data_path", type=str, default=f"{ROOT_PATH}/dataset/test_data_{VERSION}{TAG}.hdf5")
parser.add_argument('--ep', dest="epoch", type=int, default=30)
parser.add_argument('--bs', dest="batch_size", type=int, default=256)
parser.add_argument('--lr', dest="lr", type=float, default=0.001)
parser.add_argument('--use-graph', dest="use_graph", default=False, action='store_true')
parser.add_argument('--add-edge', dest="add_edge", default=False, action='store_true')
parser.add_argument('--eval', dest="eval", default=False, action='store_true')
parser.add_argument('--version', dest="version", type=str, default="test")
parser.add_argument('--pretrain', dest="pretrain", type=str, default="./param/policy_with_graph_lstm.pt")
parser.add_argument("--use-rand", dest="use_rand", type=int, default=-1, help="-1: ML, 0: random, 1: biased")
parser.add_argument("--num-user-state", dest="num_user_state", type=int, default=100, help="number of user state")
parser.add_argument("--topk", dest="topk", type=int, default=-1, help="topk=-1: calculate accuracy; topk>1, calculate hit rate@topk")
args = parser.parse_args()


logging.basicConfig(
    filename="./logs/log_train_regression_{}.txt".format(args.version),
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 


video_embeddings, video_graph_adj_mat, video_ids_map, num_videos, emb_dim = load_metadata(
    video_emb_path=args.video_emb_path,
    video_graph_path=args.video_graph_path,
    video_id_path=args.video_id_path,
    logger=logger
)
print(video_embeddings.shape[0])

train_loader, test_loader, val_loader = load_dataset(
    train_data_path=args.train_data_path,
    test_data_path=args.test_data_path,
    batch_size=args.batch_size,
    logger=logger
)

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


video_embeddings = torch.from_numpy(video_embeddings)

agg_video_graph = GraphAggregator(
    video_embeddings=video_embeddings, 
    emb_dim=emb_dim, 
    add_edge=args.add_edge,
    device=device
)

video_graph_embeddings = GraphEncoder(
    video_embeddings=video_embeddings, 
    emb_dim=emb_dim, 
    video_graph_adj_mat=video_graph_adj_mat, 
    aggregator=agg_video_graph,
    device=device
)

policy_net = PolicyNetRegression(
    emb_dim=emb_dim,
    hidden_dim=128,
    graph_embeddings=video_graph_embeddings,
    video_embeddings=video_embeddings,
    device=device,
    topk=args.topk,
    use_rand=args.use_rand,
    num_user_state=args.num_user_state
)


optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
best_acc = 1000
if args.eval == False:
    for ep in range(args.epoch):
        # State tracker
        stat = {
            "train_last_acc": 0, "train_last_acc_ch": 0, "train_last_count": 0, "train_loss_pos": 0, "train_loss_neg": 0, 
            "test_last_acc": 0, "test_last_acc_ch": 0, "test_last_count": 0, "test_loss_pos": 0, "test_loss_neg": 0
        }

        # Training
        stat = run_regression_epoch(model=policy_net, dataloader=train_loader, mode="train", optimizer=optimizer, ep=ep, stat=stat, logger=logger, use_graph=args.use_graph)

        # Testing
        stat = run_regression_epoch(model=policy_net, dataloader=val_loader, mode="test", optimizer=optimizer, ep=ep, stat=stat, logger=logger, use_graph=args.use_graph)

        # Save model
        if stat["test_last_acc_ch"] < best_acc:
            best_acc = stat["test_last_acc_ch"]
            torch.save(policy_net, "./param/policy_{}.pt".format(args.version))
else:
    # State tracker
    stat = {
        "train_last_acc": 0, "train_last_acc_ch": 0, "train_last_count": 0, "train_loss_pos": 0, "train_loss_neg": 0, 
        "test_last_acc": 0, "test_last_acc_ch": 0, "test_last_count": 0, "test_loss_pos": 0, "test_loss_neg": 0
    }
    policy_net.load_state_dict(torch.load(args.pretrain, map_location=device).state_dict())
    policy_net.device = device
    policy_net.topk = args.topk
    if args.use_graph:
        policy_net.video_embeddings = video_embeddings.to(device)
        policy_net.graph_embeddings.device = device
        policy_net.graph_embeddings.aggregator.device = device
    logger.info("load model")
    # Testing
    stat = run_regression_epoch(model=policy_net, dataloader=test_loader, mode="test", optimizer=optimizer, ep=0, stat=stat, logger=logger, use_graph=args.use_graph)


