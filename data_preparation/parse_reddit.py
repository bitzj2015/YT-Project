import os, json
from random import sample
from tqdm import tqdm
import ray
import subprocess

def get_automatic_captions(js: dict):
    captions = js.get('automatic_captions', {})
    if 'en' not in captions:
        return None
    return captions['en'][0]['url']


def get_subtitles(js: dict):
    subtitles = js.get('subtitles', {})
    if 'en' not in subtitles:
        return None
    return subtitles['en'][0]['url']


@ray.remote
def get_metadata(video_id_list: list):
    ret = {}
    for video_id in tqdm(video_id_list):
        try:
            url = 'https://youtube.com/watch?v=%s' % video_id
            if os.path.exists(f"/scratch/YT_dataset/metadata/{video_id}.json"):
                js = json.load(open(f"/scratch/YT_dataset/metadata/{video_id}.json", "r"))
            else:
                js = json.loads(subprocess.run(['/usr/local/bin/youtube-dl', '-J', url], stdout=subprocess.PIPE).stdout)
                with open(f"/scratch/YT_dataset/metadata/{video_id}.json", "w") as json_file:
                    json.dump(js, json_file)
            metadata = dict(
                title=js.get('title', ''),
                channel_id=js.get('channel_id', ''),
                description=js.get('description', ''),
                view_count=js.get('view_count', ''),
                average_rating=js.get("average_rating", ''),
                thumbnails=','.join([t.get('url') for t in js.get('thumbnails', '')]),
                categories=','.join(js.get('categories', [])),
                subtitles=get_subtitles(js),
                automated_captions=get_automatic_captions(js)
            )
            ret.update({video_id: metadata})

        except:
            ret.update({video_id: {}})

    return ret

PHASE = 2

if PHASE == 0:
    with open("/scratch/YT_dataset/reddit/Final_Data_for_Crawl.txt", "r") as json_file:
        data = json.load(json_file)

    count = 0 
    count_miss = 0
    daily_trace_len_stat = {}
    trace_len_stat = {}
    max_interval = 0
    max_trace_len = 0

    all_user_data = {}
    for user in tqdm(data.keys()):
        video_list = data[user]
        video_id_list = []
        video_timestamp_dict = {}
        num_user_videos = len(video_list)
        max_trace_len = max(num_user_videos, max_trace_len)
        
        if num_user_videos not in trace_len_stat.keys():
            trace_len_stat[num_user_videos] = 0
        trace_len_stat[num_user_videos] += 1
        
        if num_user_videos < 40 or num_user_videos > 20000:
            continue
        for item in video_list:
            video_url = item[0]
            video_timestamp = item[1]
            count += 1
            video_id = ""
            try:
                if "youtube.com/watch?v=" in video_url:
                    video_id = video_url.split("url=")[-1].split("&amp")[0].split("v=")[1]

                elif "youtube.com/watch/?v=" in video_url:
                    video_id = video_url.split("url=")[-1].split("&amp")[0].split("v=")[1]

                elif "youtube.com/watch?" in video_url:
                    video_id = video_url.split("&amp")[1].split("v=")[1]

                elif "youtube.com/watch/" in video_url:
                    video_id = video_url.split("url=")[-1].split("&")[0].split("/")[-1]
                    
                elif "youtube.com/v/" in video_url:
                    video_id = video_url.split("&")[0].split("?")[0].split("/")[-1]
                    
                elif "youtu.be" in video_url:
                    video_id = video_url.split("?")[0].split("/")[-1]

                # elif "youtube.com/attribution_link?" in video_url:
                #     continue
                # elif video_url.startswith("https://www.reddit.com"):
                #     continue
                # elif "youtube.com/embed" in video_url:
                #     print(video_url)
                #     continue
                # elif "youtube.com/playlist" in video_url:
                #     continue  
                else:
                    count_miss += 1
            except:
                count_miss += 1

            if video_id != "":
                video_id_list.append(video_id)
                date, time = video_timestamp.split(" ")
                if date not in video_timestamp_dict.keys():
                    video_timestamp_dict[date] = {"time": [], "video": []}
                video_timestamp_dict[date]["time"].append(time)
                video_timestamp_dict[date]["video"].append(video_id)

        for date in video_timestamp_dict.keys():
            interval = len(video_timestamp_dict[date])
            if interval > max_interval:
                max_interval = interval
            if interval not in daily_trace_len_stat.keys():
                daily_trace_len_stat[interval] = 0
            daily_trace_len_stat[interval] += 1

        all_user_data[user] = video_timestamp_dict

    print(f"Percent of missing video urls:{count_miss / count}")
    with open("../dataset/all_user_data.json", "w") as json_file:
        json.dump(all_user_data, json_file)

    with open("../dataset/daily_trace_len_stat.json", "w") as json_file:
        json.dump(daily_trace_len_stat, json_file)

    with open("../dataset/trace_len_stat.json", "w") as json_file:
        json.dump(trace_len_stat, json_file)

elif PHASE == 1:
    seq_len = 40
    with open("../dataset/all_user_data.json", "r") as json_file:
        all_user_data = json.load(json_file)
    
    sample_traces = {}
    for user in tqdm(all_user_data.keys()):
        user_video_list = []
        for date in sorted(all_user_data[user].keys()):
            user_video_list += all_user_data[user][date]["video"]
        num_user_videos = len(user_video_list)
        num_batches = num_user_videos // 40

        if num_batches > 0:
            sample_traces[user] = user_video_list[(num_batches - 1) * seq_len : num_batches * seq_len]
    
    with open("../dataset/sample_reddit_traces.json", "w") as json_file:
        json.dump(sample_traces, json_file)
    
    tmp = json.dumps(sample_traces[user])
    print(type(json.loads(tmp)))

else:
    with open("../dataset/sample_reddit_traces.json", "r") as json_file:
        sample_traces = json.load(json_file)
    
    reddit_videos = {}
    for user in sample_traces.keys():
        data = sample_traces[user]
        for video in data:
            reddit_videos[video] = {}


    # Get video list
    video_id_list = list(reddit_videos.keys())
    print(len(video_id_list))
    num_cpus = os.cpu_count()
    batch_size = len(video_id_list) // num_cpus + 1

    print("Start getting metadata for videos.")
    ray.init()
    rets = ray.get(
        [get_metadata.remote(video_id_list[i*batch_size: (i+1)*batch_size]) for i in range(num_cpus)]
    )
    ray.shutdown()
    print("Finish getting metadata for videos.")

    # Save all video metadata
    video_metadata = {} #
    # video_metadata = json.load(open(video_metadata_path, "r"))
    for ret in rets:
        video_metadata.update(ret)

    with open("../dataset/video_metadata_reddit_all.json", "w") as json_file:
        json.dump(video_metadata, json_file)