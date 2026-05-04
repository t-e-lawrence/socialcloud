#!/usr/bin/env python3
"""
Instagram network visualization — generates interactive graph + static charts.

Usage:
    python analysis/visualize.py
    python analysis/visualize.py --output ./my_output
"""

import argparse
import base64
import io
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

import community as community_louvain
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from PIL import Image

DB_PATH = Path(os.environ.get("CRAWLER_DB_PATH", "data/network.db"))
DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "data/downloads"))
TEMPLATE_PATH = Path(__file__).parent / "templates" / "network.html"
DEFAULT_OUTPUT = Path(os.environ.get("ANALYSIS_OUTPUT_DIR", "data/output"))

GENDER_COLORS = {"male": "#4A90D9", "female": "#E84393", "undetermined": "#95A5A6"}
COMMUNITY_PALETTE = [
    "#E74C3C",
    "#3498DB",
    "#2ECC71",
    "#F39C12",
    "#9B59B6",
    "#1ABC9C",
    "#E67E22",
    "#34495E",
    "#16A085",
    "#C0392B",
    "#2980B9",
    "#27AE60",
    "#F1C40F",
    "#8E44AD",
    "#D35400",
]


def load_data():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    profiles = {
        r["username"]: dict(r) for r in conn.execute("SELECT * FROM profiles WHERE crawl_status = 'crawled'").fetchall()
    }

    crawled_set = set(profiles.keys())

    edges = [
        (r["source"], r["target"])
        for r in conn.execute("SELECT source, target FROM follows").fetchall()
        if r["source"] in crawled_set and r["target"] in crawled_set
    ]

    posts = [dict(r) for r in conn.execute("SELECT * FROM posts").fetchall()]

    comments = [dict(r) for r in conn.execute("SELECT * FROM comments").fetchall()]

    conn.close()
    return profiles, edges, posts, comments


def build_graph(profiles, edges):
    G = nx.DiGraph()
    for uname, p in profiles.items():
        G.add_node(
            uname,
            display_name=p["display_name"] or "",
            gender=p["gender"] or "undetermined",
            followers=p["follower_count"] or 0,
            following=p["following_count"] or 0,
            media_count=p["media_count"] or 0,
            is_private=bool(p["is_private"]),
            depth=p["crawl_depth"] or 0,
        )
    for src, tgt in edges:
        G.add_edge(src, tgt)
    return G


# ── 1. Interactive network graph with media browser ────────────────


def _encode_thumbnail(file_path, max_size=200):
    """Resize image (or extract video frame) and return base64 data URI."""
    try:
        if file_path.suffix.lower() in (".mp4", ".webm"):
            return _encode_video_thumbnail(file_path, max_size)
        img = Image.open(file_path)
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _encode_video_thumbnail(video_path, max_size=200):
    """Extract first frame from video as a thumbnail."""
    import subprocess

    try:
        # Use ffmpeg to extract first frame
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-q:v",
                "5",
                "-",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        img = Image.open(io.BytesIO(result.stdout))
        img.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def make_interactive_graph(G, partition, comm_homogeneity, profiles, posts, output_dir):
    print("  Building node data...")
    nodes_json = []
    for node, data in G.nodes(data=True):
        comm = partition.get(node, 0)
        nodes_json.append(
            {
                "id": node,
                "label": node,
                "display_name": data.get("display_name", ""),
                "gender": data.get("gender", "undetermined"),
                "followers": data.get("followers", 0),
                "following": data.get("following", 0),
                "media_count": data.get("media_count", 0),
                "is_private": data.get("is_private", False),
                "community": comm,
                "homogeneity": comm_homogeneity.get(comm),
                "depth": data.get("depth", 0),
            }
        )

    edges_json = [{"from": s, "to": t} for s, t in G.edges()]

    # Community data
    comm_data = {}
    for comm_id in set(partition.values()):
        members = [n for n, c in partition.items() if c == comm_id]
        comm_data[comm_id] = {
            "size": len(members),
            "homogeneity": comm_homogeneity.get(comm_id),
            "members": members,
        }

    # Build per-profile data with file paths + pfp
    print("  Collecting media paths...")
    posts_by_owner = {}
    for p in posts:
        posts_by_owner.setdefault(p.get("owner", ""), []).append(p)

    profile_data = {}
    media_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4", ".webm"}
    media_count = 0
    downloads_abs = DOWNLOADS_DIR.resolve()

    for username in G.nodes():
        pdata = {
            "posts": [],
            "stories": [],
            "highlights": {},
            "following": [],
            "followed_by": [],
            "pfp_url": None,
        }

        # Profile picture URL from DB
        p = profiles.get(username, {})
        pdata["pfp_url"] = p.get("pfp_url")

        user_dir = downloads_abs / username
        if user_dir.exists():
            # Posts
            posts_dir = user_dir / "posts"
            if posts_dir.exists():
                for f in sorted(posts_dir.glob("*"))[:12]:
                    if f.suffix.lower() in media_exts:
                        is_video = f.suffix.lower() in (".mp4", ".webm")
                        pdata["posts"].append(
                            {
                                "src": f"file://{f}",
                                "is_video": is_video,
                            }
                        )
                        media_count += 1

            # Stories
            stories_dir = user_dir / "stories"
            if stories_dir.exists():
                for f in sorted(stories_dir.glob("*"))[:6]:
                    if f.suffix.lower() in media_exts:
                        is_video = f.suffix.lower() in (".mp4", ".webm")
                        pdata["stories"].append(
                            {
                                "src": f"file://{f}",
                                "is_video": is_video,
                            }
                        )
                        media_count += 1

            # Highlights
            highlights_dir = user_dir / "highlights"
            if highlights_dir.exists():
                for hl_dir in sorted(highlights_dir.iterdir())[:5]:
                    if hl_dir.is_dir():
                        hl_name = hl_dir.name.split("_", 1)[-1] if "_" in hl_dir.name else hl_dir.name
                        items = []
                        for f in sorted(hl_dir.glob("*"))[:5]:
                            if f.suffix.lower() in media_exts:
                                is_video = f.suffix.lower() in (".mp4", ".webm")
                                items.append(
                                    {
                                        "src": f"file://{f}",
                                        "is_video": is_video,
                                    }
                                )
                                media_count += 1
                        if items:
                            pdata["highlights"][hl_name] = items

        # Attach post metadata
        user_posts = posts_by_owner.get(username, [])
        for i, p in enumerate(user_posts):
            if i < len(pdata["posts"]):
                pdata["posts"][i]["caption"] = p.get("caption")
                pdata["posts"][i]["likes"] = p.get("like_count")
                pdata["posts"][i]["comments"] = p.get("comment_count")

        # Follow lists
        pdata["following"] = sorted([t for s, t in G.edges() if s == username])
        pdata["followed_by"] = sorted([s for s, t in G.edges() if t == username])

        profile_data[username] = pdata

    print(f"  Collected {media_count} media files across {len(profile_data)} profiles")

    # Load template and inject data
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (
        template.replace("{{NODES}}", json.dumps(nodes_json, ensure_ascii=False))
        .replace("{{EDGES}}", json.dumps(edges_json))
        .replace("{{PROFILES}}", json.dumps(profile_data, ensure_ascii=False))
        .replace("{{COMMUNITIES}}", json.dumps(comm_data, ensure_ascii=False))
    )

    out = output_dir / "network.html"
    out.write_text(html, encoding="utf-8")
    print(f"  → {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")


# ── 2. Gender distribution ──────────────────────────────────────────


def make_gender_pie(G, output_dir):
    genders = Counter(d.get("gender", "undetermined") for _, d in G.nodes(data=True))
    labels = list(genders.keys())
    sizes = list(genders.values())
    colors = [GENDER_COLORS.get(g, "#95A5A6") for g in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        sizes,
        labels=[f"{l} ({s})" for l, s in zip(labels, sizes)],
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )
    ax.set_title("Gender Distribution", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "gender_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 3. Degree distribution ──────────────────────────────────────────


def make_degree_dist(G, output_dir):
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(in_degrees, bins=30, ax=ax1, color="#3498DB", kde=True)
    ax1.set_title("In-Degree Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("In-degree (followed by N crawled profiles)")
    ax1.set_ylabel("Count")

    sns.histplot(out_degrees, bins=30, ax=ax2, color="#E74C3C", kde=True)
    ax2.set_title("Out-Degree Distribution", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Out-degree (follows N crawled profiles)")
    ax2.set_ylabel("Count")

    fig.suptitle("Degree Distributions", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = output_dir / "degree_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 4. Top profiles by PageRank ─────────────────────────────────────


def make_top_profiles(G, output_dir):
    pr = nx.pagerank(G)
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]

    usernames = [u for u, _ in top]
    scores = [s for _, s in top]
    colors = [GENDER_COLORS.get(G.nodes[u].get("gender", "undetermined"), "#95A5A6") for u in usernames]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(usernames)), scores, color=colors)
    ax.set_yticks(range(len(usernames)))
    ax.set_yticklabels([f"@{u}" for u in usernames], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("PageRank Score", fontsize=12)
    ax.set_title("Top 15 Profiles by PageRank", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=c, label=g) for g, c in GENDER_COLORS.items()]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    out = output_dir / "top_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 5. Community structure ──────────────────────────────────────────


def make_communities(G, partition, output_dir):
    U = G.to_undirected()
    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, U)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(U, k=0.8, iterations=80, seed=42)

    # Draw edges
    nx.draw_networkx_edges(U, pos, alpha=0.05, width=0.3, ax=ax)

    # Draw nodes colored by community
    for comm_id in range(n_communities):
        members = [n for n, c in partition.items() if c == comm_id]
        color = COMMUNITY_PALETTE[comm_id % len(COMMUNITY_PALETTE)]
        sizes = [20 + 15 * (G.nodes[n].get("followers", 0) / 1000) ** 0.5 for n in members]
        nx.draw_networkx_nodes(
            U,
            pos,
            nodelist=members,
            node_color=color,
            node_size=sizes,
            alpha=0.8,
            ax=ax,
        )

    # Label top nodes
    pr = nx.pagerank(G)
    top_nodes = sorted(pr, key=pr.get, reverse=True)[:20]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(U, pos, labels, font_size=7, font_color="black", font_weight="bold", ax=ax)

    ax.set_title(
        f"Community Structure (Louvain) — {n_communities} communities, modularity={modularity:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    # Community legend
    comm_sizes = Counter(partition.values())
    legend_text = "\n".join(f"C{cid}: {sz} members" for cid, sz in comm_sizes.most_common(10))
    ax.text(
        0.02,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    out = output_dir / "communities.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── 6. Engagement analysis ──────────────────────────────────────────


def make_engagement(G, posts, output_dir):
    # Average likes per owner
    owner_likes = {}
    for p in posts:
        owner = p.get("owner", "")
        likes = p.get("like_count") or 0
        owner_likes.setdefault(owner, []).append(likes)

    data = []
    for owner, likes_list in owner_likes.items():
        if owner in G.nodes:
            avg_likes = sum(likes_list) / len(likes_list)
            followers = G.nodes[owner].get("followers", 0)
            gender = G.nodes[owner].get("gender", "undetermined")
            data.append(
                {
                    "username": owner,
                    "followers": followers,
                    "avg_likes": avg_likes,
                    "gender": gender,
                }
            )

    if not data:
        print("  → skipped (no post data)")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for gender, color in GENDER_COLORS.items():
        subset = [d for d in data if d["gender"] == gender]
        if subset:
            ax.scatter(
                [d["followers"] for d in subset],
                [d["avg_likes"] for d in subset],
                c=color,
                label=gender,
                alpha=0.7,
                s=50,
                edgecolors="white",
                linewidth=0.5,
            )

    # Label outliers
    for d in sorted(data, key=lambda x: x["avg_likes"], reverse=True)[:5]:
        ax.annotate(f"@{d['username']}", (d["followers"], d["avg_likes"]), fontsize=7, alpha=0.8)

    ax.set_xlabel("Followers", fontsize=12)
    ax.set_ylabel("Avg Likes per Post", fontsize=12)
    ax.set_title("Engagement: Followers vs Avg Likes", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "engagement.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


# ── Main ────────────────────────────────────────────────────────────

import sys

# Ensure project root is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Instagram network visualization")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    profiles, edges, posts, comments = load_data()
    print(f"  {len(profiles)} profiles, {len(edges)} edges, {len(posts)} posts, {len(comments)} comments")

    print("Building graph...")
    G = build_graph(profiles, edges)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Running Louvain community detection...")
    U = G.to_undirected()
    partition = community_louvain.best_partition(U)
    n_comm = len(set(partition.values()))
    print(f"  {n_comm} communities found")

    print("Computing community homogeneity...")
    comm_homogeneity = {}
    try:
        try:
            from socialcloud.analysis.embeddings import (
                _get_db,
                get_all_usernames_with_embeddings,
            )
            from socialcloud.analysis.communities import (
                cluster_semantic_homogeneity,
            )
        except ImportError:
            from embeddings import (
                _get_db,
                get_all_usernames_with_embeddings,
            )
            from communities import cluster_semantic_homogeneity
        conn = _get_db()
        users_with_emb = set(get_all_usernames_with_embeddings(conn))
        for comm_id in set(partition.values()):
            members = [n for n, c in partition.items() if c == comm_id]
            members_with_emb = [m for m in members if m in users_with_emb]
            h = cluster_semantic_homogeneity(members_with_emb, conn)
            comm_homogeneity[comm_id] = h
        conn.close()
        print(f"  Computed for {len(comm_homogeneity)} communities")
    except Exception as e:
        print(f"  Skipped (no embeddings): {e}")

    print("\nGenerating visualizations:")

    print("1. Interactive network graph with media browser...")
    make_interactive_graph(G, partition, comm_homogeneity, profiles, posts, output_dir)

    print("2. Gender distribution...")
    make_gender_pie(G, output_dir)

    print("3. Degree distribution...")
    make_degree_dist(G, output_dir)

    print("4. Top profiles by PageRank...")
    make_top_profiles(G, output_dir)

    print("5. Community structure...")
    make_communities(G, partition, output_dir)

    print("6. Engagement analysis...")
    make_engagement(G, posts, output_dir)

    print(f"\nDone! All outputs in {output_dir}/")


if __name__ == "__main__":
    main()
