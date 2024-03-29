from selenium.webdriver import Chrome, ChromeOptions, Firefox, FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from time import sleep
import re
import requests
from pyvirtualdisplay import Display

class Video:
    def __init__(self, elem, url):
        self.elem = elem
        self.url = url
        try:
            self.videoId = re.search(r'\?v=(.*)?$', url).group(1).split('&')[0]
        except:
            self.videoId = url
   
    def get_metadata(self):
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getMetadata/%s' % self.videoId)
        if r.status_code == 200:
            self.metadata = r.json()
    
    def get_slant(self):
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getSlant/%s' % self.videoId)
        if r.status_code == 200:
            js = r.json()
            self.score = js.get('slant', None)
        else:
            self.score = None
            
    def get_comments(self):
        r = requests.get('https://rostam.idav.ucdavis.edu/noyce/getComments/%s' % self.videoId)
        if r.status_code == 200:
            js = r.json()
            self.comments = js.get('comments', [])
        else:
            self.comments = []


class YTDriver:

    def __init__(self, browser='chrome', profile_dir=None, use_virtual_display=False, headless=False, verbose=False, logger=None):
        self.logger = logger
        if use_virtual_display:
            self.__log("Launching virtual display")
            Display(size=(1920,1080)).start()
            self.__log("Virtual display launched")

        if browser == 'chrome':
            self.driver = self.__init_chrome(profile_dir, headless)
        elif browser == 'firefox':
            self.driver = self.__init_firefox(profile_dir, headless)
        else:
            raise Exception("Invalid browser", browser)

        self.driver.set_page_load_timeout(30)
        self.verbose = verbose

    def close(self):
        self.driver.close()

    def get_homepage(self, scroll_times=0, use_url=False):
        # try to find the youtube icon
        max_trial = 0
        while max_trial <= 3:
            if max_trial == 0:
                if not use_url:
                    try:
                        self.__log('clicking yt icon')
                        self.driver.find_element_by_id('logo').click()
                    except:
                        self.__log('getting via url')
                        self.driver.get('https://www.youtube.com')
                else:
                    self.__log('getting via url')
                    self.driver.get('https://www.youtube.com')
            else:
                self.__log('getting via url')
                self.driver.get('https://www.youtube.com')
                
            homepage = []
            sleep(1)
            self.save_screenshot("test.png")

            try:
                # scroll page to load more results
                for _ in range(scroll_times):
                    self.driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
                    sleep(0.2)

                # collect video-like tags from homepage
                videos = self.driver.find_elements_by_xpath('//div[@id="contents"]/ytd-rich-item-renderer')

                # identify actual videos from tags
                for video in videos:
                    a = video.find_elements_by_tag_name('a')[0]
                    href = a.get_attribute('href')
                    if href is not None and href.startswith('https://www.youtube.com/watch?'):
                        homepage.append(Video(a, href))

            except Exception as e:
                self.__log(e)

            if len(homepage) > 0:
                return homepage
            else:
                max_trial += 1
                self.__log("Get 0 homepage video, so try again...")
        return homepage
    
    def play_video(self, video, duration=5, dislike=False):
        # this function returns when the video starts playing
        try:
            ret = self.__click_video(video)
            if not ret:
                self.__log("Fail to ply video, return...")
                return
            self.__click_play_button()
            self.__handle_ads()
            self.__clear_prompts()
            sleep(duration)
            if dislike:
                self.__click_dislike_button()
            # self.__click_stop_button()
        except Exception as e:
            self.__log(e)

    def get_recommendations(self, scroll_times=5):
        # wait for recommendations to appear
        try:
            for _ in range(scroll_times):
                self.driver.execute_script("window.scrollBy(0,500)","")
                sleep(0.5)
            # self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            elems = WebDriverWait(self.driver, 30).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, 'ytd-compact-video-renderer'))
            )
            return [Video(elem, elem.find_elements_by_tag_name('a')[0].get_attribute('href')) for elem in elems]
        except Exception as e:
            self.__log(e)
            return []
        # recommended videos array
        
    def save_screenshot(self, filename):
        return self.driver.save_screenshot(filename)

    ## helper methods
    def __log(self, message):
        if self.verbose:
            self.logger.info(message)

    def __click_video(self, video):
        self.tab_restart_browser()
        if type(video) == Video:
            self.__log(f"Start watching video {video.videoId}...")
            try:
                # try to click the element using selenium
                self.__log("Clicking element via Selenium...")
                video.elem.click()
                return True
            except Exception as e:
                self.__log(e)
                try:
                    # try to click the element using javascript
                    self.__log("Failed. Clicking via Javascript...")
                    self.driver.execute_script('arguments[0].click()', video.elem)
                    return True
                except Exception as e:
                    # js click failed, just open the video url
                    self.__log(e)
                    try:
                        self.__log(f"Failed. Loading video URL {video.url}...")
                        self.driver.get(video.url)
                        return True
                    except Exception as e:
                        self.__log(e)
                        return False

        elif type(video) == str:
            self.__log(f"Start watching video {video}...")
            self.driver.get(f"https://www.youtube.com/results?search_query={video}")
            try:
                # elems = WebDriverWait(self.driver, 15).until(
                #     EC.presence_of_all_elements_located((By.TAG_NAME, 'ytd-video-renderer'))
                # )
                elems = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_all_elements_located((By.XPATH, "//yt-formatted-string[@class='style-scope ytd-video-renderer']"))
                )
                elem = elems[0]
                elem.click()
                return True
            except:
                self.__log(f"Video {video} is not found...")
                return False
        else:
            raise ValueError('Unsupported video parameter!')

    def close_other_windows(self):
        """
        close all open pop-up windows and tabs other than the current one
        """
        main_handle = self.driver.current_window_handle
        windows = self.driver.window_handles
        if len(windows) > 1:
            for window in windows:
                if window != main_handle:
                    self.driver.switch_to.window(window)
                    self.driver.close()
            self.driver.switch_to.window(main_handle)

    def tab_restart_browser(self):
        self.close_other_windows()

        if self.driver.current_url.lower() == "about:blank":
            return

        # Create a new window.  Note that it is not practical to use
        # noopener here, as we would then be forced to specify a bunch of
        # other "features" that we don't know whether they are on or off.
        # Closing the old window will kill the opener anyway.
        self.driver.execute_script("window.open('')")

        # This closes the _old_ window, and does _not_ switch to the new one.
        self.driver.close()

        # The only remaining window handle will be for the new window;
        # switch to it.
        assert len(self.driver.window_handles) == 1
        self.driver.switch_to.window(self.driver.window_handles[0])

    def __click_dislike_button(self):
        el = self.driver.find_element(By.CSS_SELECTOR, '[aria-label="Dislike this video"]')
        el.click()

    def __click_play_button(self):
        try:
            playBtn = self.driver.find_elements_by_class_name('ytp-play-button')
            if 'Play' in playBtn[0].get_attribute('title'):
                playBtn[0].click()
                self.__log("succeed to play")
        except:
            pass

    def __click_stop_button(self):
        try:
            playBtn = self.driver.find_elements_by_class_name('ytp-play-button')
            if 'Pause' in playBtn[0].get_attribute('title'):
                playBtn[0].click()
        except:
            pass

    def __handle_ads(self):
        # handle multiple ads
        while True:
            sleep(1)

            # check if ad is being shown
            preview = self.driver.find_elements_by_class_name('ytp-ad-preview-container')
            if len(preview) == 0:
                self.__log('Ad not detected')
                # ad is not shown, return
                return

            self.__log('Ad detected')
            
            sleep(1)
            preview = preview[0]
            # an ad is being shown
            # grab preview text to determine ad type
            text = preview.text.replace('\n', ' ')
            wait = 0
            if 'after ad' in text:
                # unskippable ad, grab ad length
                length = self.driver.find_elements_by_class_name('ytp-ad-duration-remaining')[0].text
                wait = time2seconds(length)
                self.__log('Unskippable ad. Waiting %d seconds...' % wait)
            elif 'begin in' in text or 'end in' in text:
                # short ad
                wait = int(text.split()[-1])
                self.__log('Short ad. Waiting for %d seconds...' % wait)
            else:
                # skippable ad, grab time before skippable
                wait = int(text)
                self.__log('Skippable ad. Skipping after %d seconds...' % wait)

            # wait for ad to finish
            sleep(wait)

            # click skip button if available
            skip = self.driver.find_elements_by_class_name('ytp-ad-skip-button-container')
            if len(skip) > 0:
                skip[0].click()

    def __clear_prompts(self):
        try:
            sleep(1)
            self.driver.find_element_by_xpath('/html/body/ytd-app/ytd-popup-container/tp-yt-iron-dropdown/div/yt-tooltip-renderer/div[2]/div[1]/yt-button-renderer/a/tp-yt-paper-button/yt-formatted-string').click()
        except:
            pass
    
    def __init_chrome(self, profile_dir, headless):
        options = ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--window-size=1920,1080')
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument('--disable-application-cache')
        options.add_argument('--disable-gpu')
        options.add_argument("--disable-dev-shm-usage")
        if profile_dir is not None:
            options.add_argument('--user-data-dir=%s' % profile_dir)
        if headless:
            options.add_argument('--headless')

        return Chrome(options=options)

    def __init_firefox(self, profile_dir, headless):
        options = FirefoxOptions()
        options.add_argument('--window-size=1920,1080')
        if profile_dir is not None:
            pass
        if headless:
            options.add_argument('--headless')

        return Firefox(options=options, executable_path="/usr/local/bin/geckodriver", log_path="/home/user/Desktop/crawls/geckodriver.log")

def time2seconds(s):
    s = s.split(':')
    s.reverse()
    wait = 0
    factor = 1
    for t in s:
        wait += int(t) * factor
        factor *= 60
    return wait