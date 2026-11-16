# whenframe in a container: docker build -t whenframe . (see docs/TUTORIAL.md, "On a home server")
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY whenframe ./whenframe
RUN pip install --no-cache-dir ".[fast]" && rm -rf /src

# the index and the downloaded model live in /data, the videos are mounted read-only in /videos
ENV WHENFRAME_LIBRARY=/data/library WHENFRAME_CACHE=/data/cache PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8765
WORKDIR /data
ENTRYPOINT ["whenframe"]
CMD ["serve", "--host", "0.0.0.0", "--watch", "/videos"]
