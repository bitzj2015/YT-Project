from YTDriver import YTDriver

# load the driver
driver = YTDriver(browser='firefox', verbose=True)

# get youtube homepage
videos = driver.get_homepage() * 10

driver.play_video(videos[0], 0)

for video in driver.get_recommendations():
  video.get_metadata()
  print(video.metadata['title'])
  
# close driver
driver.close()
