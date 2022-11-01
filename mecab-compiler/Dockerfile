FROM python:3.8
LABEL MAINTAINER=masatakashiwagi

# Set some environment variables.
# PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly.

ENV PYTHONUNBUFFERED=TRUE

# PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc files which
# are unnecessary in this case.
ENV PYTHONDONTWRITEBYTECODE=TRUE

# DEBIAN_FRONTEND prevent from stoping docker build with tzdata
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && apt-get install -y --no-install-recommends \
        curl \
        sudo \
        bash \
        nkf \
        mecab \
        libmecab-dev \
        mecab-ipadic \
        mecab-ipadic-utf8 \
        git \
        wget \
        g++ \
        make \
        cmake \
        vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# nodejsの導入
RUN curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash - \
    && sudo apt-get install -y nodejs

# ipadic-neologdの導入
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && bin/install-mecab-ipadic-neologd -n -y

# Locale Japanese
ENV LC_ALL=ja_JP.UTF-8
# Timezone jst
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
# mecabrcが見つからないと怒られてるので、環境変数MECABRCにパスを通す。(https://qiita.com/NLPingu/items/6f794635c4ac35889da6)
ENV MECABRC=/etc/mecabrc

COPY requirements.txt /tmp/requirements.lock
RUN python3 -m pip install -U pip && \
    python3 -m pip install -r /tmp/requirements.lock && \
    rm /tmp/requirements.lock && \
    rm -rf /root/.cache

# Set up the program in the image
ENV PROGRAM_DIR=/opt/program
COPY app $PROGRAM_DIR
WORKDIR $PROGRAM_DIR
ENV PATH="/opt/program:${PATH}"
ENV PYTHONPATH /opt/program/

CMD ["python3"]