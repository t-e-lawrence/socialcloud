#!/usr/bin/env python3
"""
Analysis CLI.

Usage:
    python -m analysis encode         # encode all content → embeddings table
    python -m analysis metrics        # per-user variance metrics
    python -m analysis distances      # pairwise Wasserstein distances
    python -m analysis communities    # within/between cluster semantic analysis
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Instagram network content analysis")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("encode", help="Encode all content into CLIP embeddings")
    sub.add_parser("metrics", help="Compute per-user cloud metrics")
    sub.add_parser("distances", help="Compute pairwise Wasserstein distances")
    sub.add_parser("communities", help="Community semantic analysis")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "encode":
        from socialcloud.analysis.embeddings import encode_all

        encode_all()
    elif args.command == "metrics":
        from socialcloud.analysis.embeddings import compute_all_metrics

        compute_all_metrics()
    elif args.command == "distances":
        from socialcloud.analysis.embeddings import compute_pairwise_distances

        compute_pairwise_distances()
    elif args.command == "communities":
        from socialcloud.analysis.communities import analyze_communities

        analyze_communities()


if __name__ == "__main__":
    main()
