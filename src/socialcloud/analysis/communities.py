"""
Community semantic analysis — within-cluster homogeneity and between-cluster similarity
using CLIP embedding clouds and Louvain structural communities.
"""

import logging
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance_nd

from socialcloud.analysis.embeddings import _get_db, get_embeddings, get_all_usernames_with_embeddings

logger = logging.getLogger(__name__)

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


def build_graph() -> nx.DiGraph:
    """Build NetworkX graph from the follows table."""
    conn = _get_db()
    G = nx.DiGraph()

    profiles = conn.execute(
        "SELECT username, display_name, gender, is_private, crawl_depth FROM profiles WHERE crawl_status = 'crawled'"
    ).fetchall()
    for p in profiles:
        G.add_node(
            p["username"],
            display_name=p["display_name"] or "",
            gender=p["gender"] or "?",
            is_private=bool(p["is_private"]),
            depth=p["crawl_depth"],
        )

    edges = conn.execute(
        "SELECT source, target FROM follows WHERE source IN (SELECT username FROM profiles WHERE crawl_status = 'crawled')"
    ).fetchall()
    for e in edges:
        if e["source"] in G and e["target"] in G:
            G.add_edge(e["source"], e["target"])

    conn.close()
    return G


def get_louvain_communities(G: nx.DiGraph) -> dict[str, int]:
    """Run Louvain on undirected version, return username -> community_id."""
    if community_louvain is None:
        raise ImportError("pip install python-louvain")
    U = G.to_undirected()
    return community_louvain.best_partition(U)


def cluster_semantic_homogeneity(members: list[str], conn) -> float | None:
    """Mean pairwise Wasserstein distance within a community. Lower = more homogeneous."""
    user_vecs = {}
    for u in members:
        vecs = get_embeddings(conn, u)
        if len(vecs) >= 2:
            user_vecs[u] = vecs

    if len(user_vecs) < 2:
        return None

    distances = []
    for a, b in combinations(user_vecs.keys(), 2):
        distances.append(wasserstein_distance_nd(user_vecs[a], user_vecs[b]))

    return float(np.mean(distances))


def cross_cluster_distance(members_a: list[str], members_b: list[str], conn) -> float | None:
    """Wasserstein distance between two communities' aggregate embedding clouds."""
    vecs_a = []
    for u in members_a:
        v = get_embeddings(conn, u)
        if len(v) > 0:
            vecs_a.append(v)
    vecs_b = []
    for u in members_b:
        v = get_embeddings(conn, u)
        if len(v) > 0:
            vecs_b.append(v)

    if not vecs_a or not vecs_b:
        return None

    all_a = np.concatenate(vecs_a)
    all_b = np.concatenate(vecs_b)

    if len(all_a) < 2 or len(all_b) < 2:
        return None

    return float(wasserstein_distance_nd(all_a, all_b))


def analyze_communities():
    """Full community semantic analysis: within-cluster homogeneity + between-cluster distances."""
    G = build_graph()
    partition = get_louvain_communities(G)
    modularity = community_louvain.modularity(partition, G.to_undirected())

    # Group members by community
    communities = {}
    for user, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(user)

    # Sort by size
    sorted_comms = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)

    conn = _get_db()
    users_with_embeddings = set(get_all_usernames_with_embeddings(conn))

    print(f"\n{'=' * 70}")
    print("Community Semantic Analysis")
    print(f"{'=' * 70}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Louvain: {len(communities)} communities, modularity={modularity:.3f}")

    # Within-cluster homogeneity
    print("\n--- Within-Cluster Semantic Homogeneity ---")
    print("(lower = more homogeneous = genuine tribe)")
    print()

    comm_scores = []
    for comm_id, members in sorted_comms:
        members_with_emb = [m for m in members if m in users_with_embeddings]
        homogeneity = cluster_semantic_homogeneity(members_with_emb, conn)
        genders = {}
        for m in members:
            g = G.nodes[m].get("gender", "?")
            genders[g] = genders.get(g, 0) + 1
        gender_str = ", ".join(f"{g}:{c}" for g, c in sorted(genders.items(), key=lambda x: -x[1]))

        h_str = f"{homogeneity:.4f}" if homogeneity is not None else "N/A"
        print(
            f"  Community {comm_id} ({len(members)} members, {len(members_with_emb)} with embeddings) — homogeneity={h_str}"
        )
        print(f"    Gender: {gender_str}")
        sample = members[:5]
        for m in sample:
            name = G.nodes[m].get("display_name", "")
            print(f"      {m} ({name})")
        if len(members) > 5:
            print(f"      ... +{len(members) - 5} more")
        print()

        if homogeneity is not None:
            comm_scores.append((comm_id, homogeneity, len(members)))

    # Between-cluster distances
    print("\n--- Between-Cluster Semantic Distances ---")
    print("(lower = latent community = should know each other but don't)")
    print()

    # Only compare communities with enough embeddings
    viable = [
        (cid, members) for cid, members in sorted_comms if sum(1 for m in members if m in users_with_embeddings) >= 3
    ]

    cross_distances = []
    for (cid_a, members_a), (cid_b, members_b) in combinations(viable, 2):
        emb_a = [m for m in members_a if m in users_with_embeddings]
        emb_b = [m for m in members_b if m in users_with_embeddings]
        dist = cross_cluster_distance(emb_a, emb_b, conn)
        if dist is not None:
            cross_distances.append((cid_a, cid_b, dist, len(members_a), len(members_b)))

    cross_distances.sort(key=lambda x: x[2])

    if cross_distances:
        print("  Most SIMILAR clusters (potential latent communities):")
        for cid_a, cid_b, dist, size_a, size_b in cross_distances[:5]:
            print(f"    Community {cid_a} ({size_a}) ↔ Community {cid_b} ({size_b})  dist={dist:.4f}")

        print("\n  Most DIFFERENT clusters:")
        for cid_a, cid_b, dist, size_a, size_b in cross_distances[-5:]:
            print(f"    Community {cid_a} ({size_a}) ↔ Community {cid_b} ({size_b})  dist={dist:.4f}")
    else:
        print("  Not enough embedding data for cross-cluster analysis.")

    conn.close()
