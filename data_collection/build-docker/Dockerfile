FROM ubuntu:18.04
RUN apt-get -qqy update --no-install-recommends && \
    apt-get -qqy install sudo python3-pip firefox wget
    
COPY crawler.py /opt/crawler.py
COPY crawler_eval.py /opt/crawler_eval.py
COPY crawler_attack.py /opt/crawler_attack.py
COPY crawler_reload.py /opt/crawler_reload.py
COPY YTDriver.py /opt/YTDriver.py
COPY requirements.txt /opt/requirements.txt
    
RUN useradd user \
         --shell /bin/bash  \
         --create-home \
  && usermod -a -G sudo user \
  && echo 'ALL ALL = (ALL) NOPASSWD: ALL' >> /etc/sudoers \
  && echo 'user:secret' | chpasswd

USER user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN sudo chown -R user:user /home/user && \
    sudo mkdir /home/user/Desktop && \
    sudo mkdir /home/user/Desktop/crawls && \
    sudo chown -R user:user /home/user && \
    sudo chown -R user:user /opt/crawler.py && \
    sudo chmod -R 777 /home/user/Desktop && \
    sudo chmod -R 777 /home/user/Desktop/crawls && \
    sudo pip3 install -r /opt/requirements.txt && \
    sudo wget https://github.com/mozilla/geckodriver/releases/download/v0.27.0/geckodriver-v0.27.0-linux64.tar.gz && \
    sudo tar -xvzf geckodriver-v0.27.0-linux64.tar.gz -C /usr/local/bin && \
    sudo chmod +x /usr/local/bin/geckodriver && \
    sudo chown -R user:user /opt
