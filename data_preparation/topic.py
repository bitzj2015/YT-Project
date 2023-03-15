import os
import h5py
import json
import numpy as np
from constants import *
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings


VERSION = "_realuser"
VERSION = "_40"
VERSION = "_realuser_0.2_test_reload"
# VERSION = "_latest_joint_cate_010"
VERSION = "_v1_binary_0.2_test"
VERSION = "_realuser_0.2_test"
# VERSION = "_0.5_v2_kldiv_0.5_test_0.2_test"
# VERSION = "_0.5_v2_kldiv_0.5_test"
# VERSION = "_realuser"
# VERSION = "_reddit_40_new"
# VERSION = "_v2_kldiv_sensitive"
# VERSION = "_0.2_v2_kldiv_reddit2_test"
VERSION = "_realuser"
VERSION = "_0.2_v2_kldiv_pbooster_3_new_v2"
VERSION = "_0.5_v2_kldiv_pbooster_0.5_3_new_v2"
VERSION = "_0.7_v2_kldiv_feb_0.7_0_new_v2"
VERSION = "_0.2_v2_kldiv_pbooster_reddit_0.2_3_new_v2"
VERSION = "_0.5_realuser_0_new_v2"
VERSION = "_0.2_realuser_pbooster_3_new_v2"



saved_path = "../"

for phase in ["get_tag", "get_embed", "get_sim", "get_map"]:
    if phase == "get_tag":
        with open(f"{root_path}/dataset/video_metadata{VERSION}_new.json", "r") as json_file:
            video_metadata = json.load(json_file)
        # print(cate_ids)

        tag_dict = {}
        cnt = 0
        for video_id in video_metadata.keys():
            try:
                tags = video_metadata[video_id]["tags"].split(",")
                if len(tags) == 0:
                    cnt += 1
                for tag in tags:
                    if tag == "" or tag == " " or len(tag.replace(" ", "")) == 0:
                        continue
                    if tag not in tag_dict.keys():
                        tag_dict[tag] = 0
                    tag_dict[tag] += 1
            except:
                continue

        tag_dict = {k: v for k, v in sorted(tag_dict.items(), key=lambda item: item[1], reverse=True)}
        with open (f"{saved_path}/dataset/topic/tags{VERSION}.json", "w") as json_file:
            json.dump(tag_dict, json_file)
    
    elif phase == "get_embed":
        # initialize the word embeddings
        glove_embedding = WordEmbeddings('glove')

        # initialize the document embeddings, mode = mean
        document_embeddings = DocumentPoolEmbeddings([glove_embedding])

        with open(f"{saved_path}/dataset/topic/class_ads.json", "r") as json_file:
            classes = json.load(json_file)

        try:
            del classes[""]
            del classes["name"]
        except:
            print("Continue")
        # create an example sentence
        class_emb = [Sentence(c.replace("& ", "").replace("-", "")) for c in classes.keys()]

        # embed the sentence with our document embedding
        document_embeddings.embed(class_emb)
        class_embs = [c.embedding.tolist() for c in class_emb]
        class_embs = np.array(class_embs)
        print(class_embs.shape)

        with open(f"{saved_path}/dataset/topic/tags{VERSION}.json", "r") as json_file:
            tags = json.load(json_file)

        tmp = list(tags.keys())

        tags_embs = []
        for c in tmp:
            try:
                sentence_c = Sentence(c)
                document_embeddings.embed(sentence_c)
                tags_embs.append(sentence_c.embedding.tolist())
            except:
                del tags[c]
                print(c)

        tags_embs = np.array(tags_embs)
        assert len(tags.keys()) == tags_embs.shape[0]

        with open (f"{saved_path}/dataset/topic/tags{VERSION}.json", "w") as json_file:
            json.dump(tags, json_file)

        hf = h5py.File(f"{saved_path}/dataset/topic/data{VERSION}2.hdf5", 'w')
        hf.create_dataset('classes', data=class_embs)
        hf.create_dataset('tags', data=tags_embs)
        hf.close()

    elif phase == "get_sim":
        hf = h5py.File(f"{saved_path}/dataset/topic/data{VERSION}2.hdf5", 'r')
        class_embs = np.array(hf['classes'][:])
        tags_embs = np.array(hf['tags'][:])
        hf.close()

        sim_mat = np.matmul(tags_embs, class_embs.transpose()) / \
            (np.linalg.norm(tags_embs, axis=-1).reshape(-1,1) * np.linalg.norm(class_embs, axis=-1).reshape(1,-1) + 1e-7)
        sim_ret = np.argmax(sim_mat, axis=-1)
        print(sim_mat.shape, sim_ret.shape)

        hf = h5py.File(f"{saved_path}/dataset/topic/data{VERSION}2.hdf5", 'a')
        hf.create_dataset('sim_mat', data=sim_mat)
        hf.create_dataset('sim_ret', data=sim_ret)
        hf.close()

    elif phase == "get_map":
        hf = h5py.File(f"{saved_path}/dataset/topic/data{VERSION}2.hdf5", 'r')
        sim_mat = np.array(hf['sim_mat'][:])
        sim_ret = np.array(hf['sim_ret'][:])
        sim_max = np.max(sim_mat, axis=-1)
        hf.close()

        with open(f"{saved_path}/dataset/topic/class_ads.json", "r") as json_file:
            classes = json.load(json_file)
        try:
            del classes[""]
            del classes["name"]
        except:
            print("Continue")

        classes_list = list(classes.keys())
        class2id = dict(zip(classes_list, [i for i in range(len(classes_list))]))

        with open(f"{saved_path}/dataset/topic/tags{VERSION}.json", "r") as json_file:
            tags = json.load(json_file)

        assert len(tags) == len(sim_max)
        tag2class = {}
        class_list = list(classes.keys())
        tag_list = list(tags.keys())

        for i in range(len(tag_list)):
            tag2class[tag_list[i]] = (class_list[sim_ret[i]], sim_max[i])

        with open(f"{saved_path}/dataset/topic/tag2class{VERSION}2.json", "w") as json_file:
            json.dump(tag2class, json_file)

        with open(f"{saved_path}/dataset/topic/class2id2.json", "w") as json_file:
            json.dump(class2id, json_file)

    else:
        print("No such operation!")