#=============================================================
# Dockerfile for OpenWPM
# See README.md for build & use instructions
#=============================================================

FROM ubuntu:18.04

#=============================================================
# Packages required for container setup
#=============================================================

RUN apt-get -qqy update --no-install-recommends && \
    apt-get -qqy install sudo python3-pip
    
COPY download_comment.py /opt/download_comment.py
    
RUN useradd user \
         --shell /bin/bash  \
         --create-home \
  && usermod -a -G sudo user \
  && echo 'ALL ALL = (ALL) NOPASSWD: ALL' >> /etc/sudoers \
  && echo 'user:secret' | chpasswd

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN sudo chown -R user:user /home/user && \
    alias ll='ls -al' && \
    sudo mkdir /home/user/Desktop && \
    sudo mkdir /home/user/Desktop/crawls && \
    sudo chown -R user:user /home/user && \
    sudo chown -R user:user /opt/crawler.py && \
    sudo chmod 777 -R /home/user/Desktop/ && \
    sudo pip3 install nltk requests-html html2text bs4 tqdm selenium && \
#     yes | sudo apt-get install chromium-chromedriver && \
    # sudo /usr/bin/python3 -m spacy download en && \
    sudo wget https://github.com/mozilla/geckodriver/releases/download/v0.27.0/geckodriver-v0.27.0-linux64.tar.gz && \
    sudo tar -xvzf geckodriver-v0.27.0-linux64.tar.gz && \
    sudo chown -R user:user /opt
