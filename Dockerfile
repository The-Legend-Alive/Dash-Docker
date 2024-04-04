FROM python:3.11-slim

COPY requirements.txt .

# Install curl
RUN apt-get update && apt-get install -y curl

# Using uv to install packages because it's fast as fuck boiiii
# https://www.youtube.com/watch?v=6E7ZGCfruaw
# https://ryxcommar.com/2024/02/15/how-to-cut-your-python-docker-builds-in-half-with-uv/
ENV VIRTUAL_ENV=/usr/local
ADD --chmod=655 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh
RUN /root/.cargo/bin/uv pip install --no-cache -r requirements.txt

COPY . ./
