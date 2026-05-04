# socialcloud

[![CI](https://github.com/t-e-lawrence/socialcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/t-e-lawrence/socialcloud/actions/workflows/ci.yml)
[![Docker](https://github.com/t-e-lawrence/socialcloud/actions/workflows/docker.yml/badge.svg)](https://github.com/t-e-lawrence/socialcloud/actions/workflows/docker.yml)

Bring-your-own-data social network analysis. Given a SQLite database of profiles, follows, comments, and posts plus the corresponding media files on disk, this package encodes content with CLIP, computes per-user embedding clouds, finds communities, and renders an interactive graph viewer.

This package does **not** scrape, crawl, or call any platform API. You collect the data however you want; this processes it.

## What this is

A data-processing pipeline that operates on a fixed schema (see [Architecture](https://github.com/t-e-lawrence/socialcloud/wiki/Architecture)). The schema is generic enough to fit any directed social network — Instagram, Twitter/X, TikTok, Mastodon, or any platform with profiles + follow edges + posts + comments.

The pipeline gives you:

- **CLIP embeddings** — every image, video first-frame, and caption encoded into a 1024-dim vector (ViT-H/14, dfn5b)
- **Per-user variance** — focused vs eclectic content identity
- **Wasserstein distance** — distributional similarity between two users' embedding clouds
- **Louvain community detection** on the follow graph, with **semantic homogeneity** scoring (mean within-community Wasserstein distance)
- **Photo matching API** — upload an image, get cosine-ranked profile matches
- **Interactive HTML viewer** — graph colored by community, side panel with profile media, drag-drop photo matching
- **CLIP gender classifier** (optional service) — zero-shot pfp + name+bio classification, no external APIs

## How it works

1. You populate `data/network.db` with profiles, follows, comments, posts (schema in the [Architecture](https://github.com/t-e-lawrence/socialcloud/wiki/Architecture) wiki page) and put media files under `data/downloads/<username>/`.
2. `analyze encode` walks the downloads, runs CLIP on every image and video first-frame, stores 1024-dim float32 vectors in the `embeddings` table.
3. `analyze metrics` computes per-user cloud statistics (centroid, variance) used for fast filtering.
4. `analyze distances` computes the full n-dimensional Wasserstein distance between every pair of user embedding clouds.
5. `analyze communities` runs Louvain on the follow graph, then scores each community's semantic homogeneity using the precomputed pairwise distances.
6. `python -m socialcloud.analysis.visualize` produces `data/output/network.html` — a self-contained interactive viewer.
7. `uvicorn socialcloud.analysis.server:app --port 8002` exposes a photo-matching endpoint backed by the embedding table.

## Installation

```bash
pip install -e .              # runtime
pip install -e ".[dev]"       # with ruff + pytest
```

This installs one console script: `analyze`. CUDA is recommended for CLIP encoding.

## Quick start

```bash
# 1. Get your data into data/network.db (your collector, your rules)
# 2. Drop media under data/downloads/<username>/

analyze encode         # populate the embeddings table
analyze metrics        # per-user cloud metrics
analyze distances      # pairwise Wasserstein distances
analyze communities    # Louvain + homogeneity scoring

python -m socialcloud.analysis.visualize
# → open data/output/network.html

uvicorn socialcloud.analysis.server:app --port 8002
# POST /match with {"image_base64": "..."} → ranked matches
```

## Optional: gender classifier

A FastAPI service that runs a zero-shot CLIP classifier over a profile picture (75% weight) and the name+bio text (25% weight). No external APIs, no quotas.

```bash
uvicorn socialcloud.gender_detection.main:app --port 8000
# POST /classify {"username": "...", "display_name": "...", "bio": "...", "pfp_base64": "..."}
```

Persists results in `data/gender_detection.db`. Useful if your collector hasn't already filled the `gender` column on `profiles`.

## Docker

`docker-compose.yml` ships two services: `gender_detection` and `analysis`. Both require an NVIDIA GPU.

```bash
docker compose up -d gender_detection
docker compose run --rm analysis encode
docker compose run --rm analysis communities
```

All persistent state lives under `./data`, mounted as a single `/data` volume per service.

### Pre-built images

CI publishes both services to the GitHub Container Registry on every push to `main`:

```bash
docker pull ghcr.io/t-e-lawrence/socialcloud/analysis:latest
docker pull ghcr.io/t-e-lawrence/socialcloud/gender-detection:latest
```

Images are also tagged with the short commit SHA and (on tagged releases) `vX.Y.Z` semver tags.

## Documentation

Full documentation lives in the [wiki](https://github.com/t-e-lawrence/socialcloud/wiki):

- [Architecture](https://github.com/t-e-lawrence/socialcloud/wiki/Architecture) — schema, package layout
- [Analysis](https://github.com/t-e-lawrence/socialcloud/wiki/Analysis) — embeddings, distances, communities, visualization
- [API Reference](https://github.com/t-e-lawrence/socialcloud/wiki/API-Reference) — gender_detection + analysis endpoints
- [Docker](https://github.com/t-e-lawrence/socialcloud/wiki/Docker) — compose usage, GPU, env vars

## License

MIT — see [LICENSE](LICENSE).
