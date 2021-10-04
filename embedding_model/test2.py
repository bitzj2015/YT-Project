from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

query_embedding = model.encode('Pelosi\'s hypocrisy is showing again. Why? Gov. Mike Huckabee gives you the LATEST Facts of the Matter!')
passage_embedding = model.encode(["Texas Senator Ted Cruz discusses Democratic lawmakers' attempt to prevent a vote on election legislation and says the Biden administration's reaction to the unrest in Cuba is 'terrible.'",
                                  "Fans cheered, waved flags and chanted “USA” as former president Donald Trump arrived at UFC 264 to watch the main event, the clash between Dustin Poirier and Conor McGregor in Las Vegas.",
                                  "Sen. John Kennedy (R-LA) slams Senate Majority Leader Chuck Schumer (D-NY) over his including the debt ceiling raise in the budget.",
                                  "At today's Senate Judiciary Committee hearing, Sen. John Kennedy (R-LA) excoriated Jennifer Sung, nominee to be United States Circuit Judge for the Ninth Circuit, over her past comments about Justice Brett Kavanaugh."])

print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
print(query_embedding.shape, passage_embedding.shape)
import json
import h5py
import random
with open("../dataset/id_videos_new.json", "r") as file:
    id_videos = json.load(file)

with open("../dataset/video_adj_list_new_w.json", "r") as file:
    video_adj_list = json.load(file)

with h5py.File("../dataset/video_embeddings_new.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:]

debug = 1
count = 0
gain_mean = 0
for video_id in video_adj_list.keys():
    related_videos = [int(key) for key in video_adj_list[video_id].keys()]
    if len(related_videos) == 0:
        count += 1
        continue
    emb = video_embeddings[int(video_id)]
    related_emb = video_embeddings[related_videos]
    sim1 = util.pytorch_cos_sim(emb, related_emb).mean()
    
    related_videos = [random.randint(0, 380000) for _ in video_adj_list[video_id].keys()]
    related_emb = video_embeddings[related_videos]
    sim2 = util.pytorch_cos_sim(emb, related_emb).mean()
    if count % 10000 == 0 and debug:
        print("Similarity:", sim1)
        print("Similarity random:", sim2)
        print("Similarity gain:", sim1-sim2)
        print("Video id:", id_videos[video_id])
        print("Related video id:", [id_videos[key] for key in video_adj_list[video_id].keys()])
    count += 1
    gain_mean += sim1-sim2

print("Average gain:", gain_mean/count)

with h5py.File("../dataset/video_embeddings_new_des.hdf5", "r") as hf_emb:
    video_embeddings = hf_emb["embeddings"][:]

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
count = 0
for video_id in video_adj_list.keys():
    related_videos = [int(key) for key in video_adj_list[video_id].keys()]
    if len(related_videos) == 0:
        count += 1
        continue
    emb = video_embeddings[int(video_id)]
    related_emb = video_embeddings[related_videos]
    sim1 = util.pytorch_cos_sim(emb, related_emb).mean()
    
    related_videos = [random.randint(0, 380000) for _ in video_adj_list[video_id].keys()]
    related_emb = video_embeddings[related_videos]
    sim2 = util.pytorch_cos_sim(emb, related_emb).mean()
    if count % 10000 == 0 and debug:
        print("Similarity:", sim1)
        print("Similarity random:", sim2)
        print("Similarity gain:", sim1-sim2)
        print("Video id:", id_videos[video_id])
        print("Related video id:", [id_videos[key] for key in video_adj_list[video_id].keys()])
    count += 1
    gain_mean += sim1-sim2

print("Average gain:", gain_mean/count)