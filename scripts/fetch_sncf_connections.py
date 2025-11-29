#!/usr/bin/env python3
"""
Fetch real SNCF railway connections from the Navitia API.

This script retrieves all routes and their stop points from the SNCF API,
then creates a connections CSV file with consecutive station pairs.
"""

import csv
import time
from pathlib import Path

import requests
from math import radians, cos, sin, asin, sqrt


API_BASE = "https://api.sncf.com/v1/coverage/sncf"
API_KEY = "974d90c9-4125-4737-83da-a756c9de6070"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "connexions.csv"


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometers between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def fetch_all_routes() -> list[dict]:
    """Fetch all routes from SNCF API."""
    routes = []
    start_page = 0
    count = 100  # Max per page

    print("Fetching routes...")
    while True:
        url = f"{API_BASE}/routes?count={count}&start_page={start_page}"
        response = requests.get(url, auth=(API_KEY, ""))
        response.raise_for_status()
        data = response.json()

        batch = data.get("routes", [])
        if not batch:
            break

        routes.extend(batch)
        total = data.get("pagination", {}).get("total_result", 0)
        print(f"  Fetched {len(routes)}/{total} routes")

        if len(routes) >= total:
            break

        start_page += 1
        time.sleep(0.1)  # Rate limiting

    return routes


def fetch_route_stop_points(route_id: str) -> list[dict]:
    """Fetch stop points for a specific route."""
    url = f"{API_BASE}/routes/{route_id}/stop_points?count=100"
    try:
        response = requests.get(url, auth=(API_KEY, ""), timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("stop_points", [])
    except Exception as e:
        print(f"  Error fetching stops for {route_id}: {e}")
        return []


def extract_connections(routes: list[dict]) -> dict:
    """
    Extract all connections from routes.

    Returns a dict of (station1, station2) -> distance to avoid duplicates.
    """
    connections = {}

    for i, route in enumerate(routes):
        route_id = route["id"]
        route_name = route.get("name", "Unknown")

        if (i + 1) % 50 == 0:
            print(f"Processing route {i + 1}/{len(routes)}: {route_name}")

        stop_points = fetch_route_stop_points(route_id)

        if len(stop_points) < 2:
            continue

        # Create connections between consecutive stops
        for j in range(len(stop_points) - 1):
            sp1 = stop_points[j]
            sp2 = stop_points[j + 1]

            name1 = sp1.get("stop_area", {}).get("name", sp1.get("name", ""))
            name2 = sp2.get("stop_area", {}).get("name", sp2.get("name", ""))

            if not name1 or not name2:
                continue

            # Get coordinates
            coord1 = sp1.get("coord", {})
            coord2 = sp2.get("coord", {})

            try:
                lat1 = float(coord1.get("lat", 0))
                lon1 = float(coord1.get("lon", 0))
                lat2 = float(coord2.get("lat", 0))
                lon2 = float(coord2.get("lon", 0))

                if lat1 == 0 or lat2 == 0:
                    continue

                distance = haversine(lat1, lon1, lat2, lon2)
            except (ValueError, TypeError):
                distance = 0

            # Use sorted tuple to avoid duplicates (A-B same as B-A)
            key = tuple(sorted([name1, name2]))

            # Keep the shortest distance if we've seen this connection before
            if key not in connections or distance < connections[key]:
                connections[key] = distance

        time.sleep(0.05)  # Rate limiting

    return connections


def save_connections(connections: dict, output_file: Path):
    """Save connections to CSV file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station1", "station2", "distance"])

        for (station1, station2), distance in sorted(connections.items()):
            writer.writerow([station1, station2, round(distance, 2)])

    print(f"\nSaved {len(connections)} connections to {output_file}")


def main():
    print("=" * 60)
    print("SNCF Connections Fetcher")
    print("=" * 60)

    # Fetch all routes
    routes = fetch_all_routes()
    print(f"\nTotal routes: {len(routes)}")

    # Extract connections
    print("\nExtracting connections from routes...")
    connections = extract_connections(routes)
    print(f"\nTotal unique connections: {len(connections)}")

    # Save to CSV
    save_connections(connections, OUTPUT_FILE)

    print("\nDone!")


if __name__ == "__main__":
    main()
