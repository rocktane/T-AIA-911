#!/usr/bin/env python3
"""
Fetch SNCF stations from the Navitia API.

Creates a gares-sncf.csv file with station names matching those in connexions.csv.
"""

import csv
import time
from pathlib import Path

import requests


API_BASE = "https://api.sncf.com/v1/coverage/sncf"
API_KEY = "974d90c9-4125-4737-83da-a756c9de6070"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "gares-sncf.csv"


def fetch_all_stop_areas() -> list[dict]:
    """Fetch all stop areas from SNCF API."""
    stop_areas = []
    start_page = 0
    count = 100

    print("Fetching stop areas...")
    while True:
        url = f"{API_BASE}/stop_areas?count={count}&start_page={start_page}"
        response = requests.get(url, auth=(API_KEY, ""), timeout=30)
        response.raise_for_status()
        data = response.json()

        batch = data.get("stop_areas", [])
        if not batch:
            break

        stop_areas.extend(batch)
        total = data.get("pagination", {}).get("total_result", 0)
        print(f"  Fetched {len(stop_areas)}/{total} stop areas")

        if len(stop_areas) >= total:
            break

        start_page += 1
        time.sleep(0.1)

    return stop_areas


def save_stations(stop_areas: list[dict], output_file: Path):
    """Save stations to CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    seen_names = set()
    stations = []

    for sa in stop_areas:
        name = sa.get("name", "").strip()
        if not name or name in seen_names:
            continue

        coord = sa.get("coord", {})
        lat = coord.get("lat", "0")
        lon = coord.get("lon", "0")

        # Skip stations without valid coordinates
        if lat == "0" and lon == "0":
            continue

        # Get UIC code if available
        codes = sa.get("codes", [])
        uic_code = ""
        for code in codes:
            if code.get("type") == "uic":
                uic_code = code.get("value", "")
                break

        seen_names.add(name)
        stations.append({
            "nom": name,
            "libellecourt": uic_code[:3] if uic_code else "",
            "lon": lon,
            "lat": lat,
            "codeinsee": uic_code,
        })

    # Sort by name
    stations.sort(key=lambda x: x["nom"])

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["nom", "libellecourt", "lon", "lat", "codeinsee"])
        writer.writeheader()
        writer.writerows(stations)

    print(f"\nSaved {len(stations)} stations to {output_file}")


def main():
    print("=" * 60)
    print("SNCF Stations Fetcher")
    print("=" * 60)

    stop_areas = fetch_all_stop_areas()
    print(f"\nTotal stop areas: {len(stop_areas)}")

    save_stations(stop_areas, OUTPUT_FILE)
    print("\nDone!")


if __name__ == "__main__":
    main()
