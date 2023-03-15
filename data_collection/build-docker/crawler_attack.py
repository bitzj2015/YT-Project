from YTDriver import YTDriver
import logging
import argparse
import json
import datetime
import uuid
import pickle
from random import choice

parser = argparse.ArgumentParser(description='run regression.')
parser.add_argument('--save-dir', dest="save_dir", type=str, default="user_0")
parser.add_argument('--video-seq', dest="video_seq", type=str, default="")
parser.add_argument('--reload', dest="reload", action='store_true')
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

if args.reload:
    driver.driver.get("https://www.youtube.com")
    cookies = pickle.load(open(f"/home/user/Desktop/crawls/{args.save_dir}/cookies.pkl", "rb"))
    for cookie in cookies:
        driver.driver.add_cookie(cookie)
        
# Get youtube initial homepage
initial_homepage_videos = driver.get_homepage()
initial_homepage_video_ids = [video.videoId for video in initial_homepage_videos]

# List for storing videos
user_trails = []
recommendation_trails = []
homepage_trails = []

# Start crawling
video_seq = json.loads(args.video_seq)
num_videos = 0
done = 0

for i in range(len(video_seq)):
    cur_video = video_seq[i]
    driver.play_video(cur_video, 30)
    if i >= len(video_seq) - 2:
        next_watch_videos = driver.get_recommendations(scroll_times=5)
    else:
        next_watch_videos = driver.get_recommendations()
    next_watch_video_ids = [video.videoId for video in next_watch_videos]
    recommendation_trails.append(next_watch_video_ids)
    if type(cur_video) == str:
        user_trails.append(cur_video)
    else:
        user_trails.append(cur_video.videoId)

# Get the final homepage recommendations
for _ in range(5):
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
results["puppetId"] = puppet_id
results["video_seq"] = video_seq

# Write results into files
with open(f"/home/user/Desktop/crawls/{args.save_dir}/trail_{puppet_id}.json", "w") as json_file:
    json.dump(results, json_file)

# Close driver
pickle.dump(driver.driver.get_cookies() , open(f"/home/user/Desktop/crawls/{args.save_dir}/cookies.pkl","wb"))
driver.close()
