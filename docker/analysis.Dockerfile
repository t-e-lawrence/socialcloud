FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir -e .
ENV CRAWLER_DB_PATH=/data/network.db
ENV DOWNLOADS_DIR=/data/downloads
ENV ANALYSIS_OUTPUT_DIR=/data/output
ENTRYPOINT ["analyze"]
