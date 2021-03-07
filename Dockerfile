FROM dmye/cpm:v0
USER root
WORKDIR /root
RUN pip install --no-cache-dir nltk jieba