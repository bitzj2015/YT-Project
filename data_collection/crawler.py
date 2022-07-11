from YTDriver import YTDriver
import logging
import argparse
import os
import json
import datetime
import uuid
import pickle
from random import choice

parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--save-dir', dest="save_dir", type=str, default="user_0")
parser.add_argument('--seed-video', dest="seed_video", type=str, default="1mchUZi_uAU")
parser.add_argument('--N', dest="N", type=int, default=10)
parser.add_argument('--L', dest="L", type=int, default=10)
args = parser.parse_args()

# Set up logging
# os.system(f"mkdir -p /home/user/Desktop/crawls/{args.save_dir}")
logging.basicConfig(
    filename=f"/home/user/Desktop/crawls/{args.save_dir}/log.txt",
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO)

# Set up metadata
start_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
puppet_id = str(uuid.uuid4())

# Load the driver
driver = YTDriver(browser='firefox', verbose=True, headless=True, logger=logger)

# Get youtube initial homepage
initial_homepage_videos = driver.get_homepage(use_url=True)
initial_homepage_video_ids = [video.videoId for video in initial_homepage_videos]

# List for storing videos
user_trails = []
recommendation_trails = []
homepage_trails = []

# Start crawling
cur_seed_video = args.seed_video
num_videos = 0
done = 0
while (1):
    for _ in range(args.L):
        driver.play_video(cur_seed_video, 30)
        next_watch_videos = driver.get_recommendations()
        next_watch_video_ids = [video.videoId for video in next_watch_videos]
        recommendation_trails.append(next_watch_video_ids)
        if type(cur_seed_video) == str:
            user_trails.append(cur_seed_video)
        else:
            user_trails.append(cur_seed_video.videoId)
        if len(next_watch_videos) > 0:
            cur_seed_video = choice(next_watch_videos)
            num_videos += 1
        else:
            num_videos += 1
            break
        if num_videos == args.N:
            done = 1
            break
    if done:
        break
    try:
        cur_seed_video = choice(driver.get_homepage())
    except:
        cur_seed_video = choice(next_watch_videos)
        driver.save_screenshot(f"/home/user/Desktop/crawls/{args.save_dir}/screenshot.png")

# Get the final homepage recommendations
for _ in range(50):
    videos = driver.get_homepage()
    homepage_trails.append([video.videoId for video in videos])

# Get end time
end_time = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

# Save results
results = {}
results["viewed"] = user_trails
results["initial_homepage"] = initial_homepage_video_ids
results["recommendation_trail"] = recommendation_trails
results["homepage"] = homepage_trails
results["start_time"] = start_time
results["end_time"] = end_time
results["L"] = args.L
results["N"] = args.N
results["puppetId"] = puppet_id
results["seedId"] = args.seed_video

# Write results into files
with open(f"/home/user/Desktop/crawls/{args.save_dir}/trail_{puppet_id}.json", "w") as json_file:
    json.dump(results, json_file)

# Close driver
pickle.dump(driver.driver.get_cookies() , open(f"/home/user/Desktop/crawls/{args.save_dir}/cookies.pkl","wb"))
driver.close()
