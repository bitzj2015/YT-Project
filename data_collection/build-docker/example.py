from YTDriver import YTDriver
import logging

# Set up logging
# os.system(f"mkdir -p /home/user/Desktop/crawls/{args.save_dir}")
logging.basicConfig(
    filename=f"log.txt",
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO
)
logger=logging.getLogger() 
logger.setLevel(logging.INFO)
# load the driver
driver = YTDriver(browser='firefox', verbose=True, logger=logger, headless=True)

# get youtube homepage
videos = driver.get_homepage() * 10

driver.play_video(videos[0], 0)

for video in driver.get_recommendations():
  print(video)
  
# close driver
driver.close()
