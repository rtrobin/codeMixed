FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

LABEL maintainer="robintang.116@gmail.com"

RUN pip install flask gunicorn gevent \
        --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple

ENV LANG C.UTF-8

COPY ./ /workspace/server
WORKDIR /workspace/server

ENV PORT=5000
ENV WORKER=4

CMD exec gunicorn -c gunicorn.py main:app
