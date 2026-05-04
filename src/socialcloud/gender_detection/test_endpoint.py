#!/usr/bin/env python3
"""
Test script for the gender detection API.

Usage:
    1. Start the server:
       GENDERAPI_KEY=<key> uvicorn gender_detection.main:app

    2. Fill in sample_profiles.csv (same directory) with rows like:
       username,display_name,pfp_path
       emily.rose,Emily Johnson,/path/to/pfp.jpg

    3. Run this script:
       python gender_detection/test_endpoint.py
       python gender_detection/test_endpoint.py --base-url http://localhost:8000
       python gender_detection/test_endpoint.py --csv path/to/profiles.csv
"""

import argparse
import base64
import csv
import sys
from pathlib import Path

import httpx

DEFAULT_CSV = Path(__file__).parent / "sample_profiles.csv"
DEFAULT_BASE = "http://localhost:8000"


def main():
    parser = argparse.ArgumentParser(description="Test the gender detection API")
    parser.add_argument("--base-url", default=DEFAULT_BASE)
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path to CSV file")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(timeout=30.0)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"No data rows in {csv_path}. Add profiles to the CSV and retry.")
        print("Expected columns: username, display_name, pfp_path")
        sys.exit(1)

    print(f"Loaded {len(rows)} profiles from {csv_path}\n")

    # POST /classify for each profile
    for row in rows:
        username = row["username"].strip()
        payload = {"username": username}
        if (row.get("display_name") or "").strip():
            payload["display_name"] = row["display_name"].strip()
        pfp_path = (row.get("pfp_path") or "").strip()
        if pfp_path:
            p = Path(pfp_path)
            if p.exists():
                payload["pfp_base64"] = base64.b64encode(p.read_bytes()).decode()
            else:
                print(f"  Warning: pfp_path not found: {pfp_path}")

        print(f"--- Classifying: {username} ---")
        resp = client.post(f"{base}/classify", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Gender: {data['gender']}  Status: {data['status']}")
            for sig, info in data["confidence_breakdown"].items():
                print(f"    {sig}: vote={info['vote']}  prob={info['raw_prob']}")
            if data.get("pending_signals"):
                print(f"  Pending: {data['pending_signals']}")
        else:
            print(f"  Error {resp.status_code}: {resp.text}")
        print()

    # GET /queue
    print("--- Queue status ---")
    resp = client.get(f"{base}/queue")
    if resp.status_code == 200:
        q = resp.json()
        print(f"  Queued: {q['queued_profiles']}")
        print(f"  GenderAPI username remaining: {q['genderapi_username_remaining']}")
        print(f"  GenderAPI name remaining:     {q['genderapi_name_remaining']}")
        print(f"  Genderize remaining:          {q['genderize_remaining']}")
        print(f"  Next reset: {q['next_reset_utc']}")
    print()

    # GET /results
    print("--- Resolved results ---")
    resp = client.get(f"{base}/results", params={"limit": 10})
    if resp.status_code == 200:
        for r in resp.json()["results"]:
            print(f"  {r['username']}: {r['gender']} ({r['status']})")
    print()


if __name__ == "__main__":
    main()
