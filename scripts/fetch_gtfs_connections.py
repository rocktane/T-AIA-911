"""
Fetch SNCF train connections from GTFS data.

Downloads the consolidated SNCF GTFS file and extracts:
- Rail connections only (no buses)
- Travel times between stations
- Train types (TGV, TER, Intercites, etc.)

Output: data/connexions.csv with columns:
    station1, station2, distance, duration_min, train_type
"""

import csv
import io
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

# GTFS URLs from SNCF Open Data
GTFS_URLS = {
    "consolidated": "https://eu.ftp.opendatasoft.com/sncf/plandata/export-opendata-sncf-gtfs.zip",
    "tgv": "https://eu.ftp.opendatasoft.com/sncf/plandata/export_gtfs_voyages.zip",
    "ter": "https://eu.ftp.opendatasoft.com/sncf/plandata/export-ter-gtfs-last.zip",
    "intercites": "https://eu.ftp.opendatasoft.com/sncf/plandata/export-intercites-gtfs-last.zip",
}

# GTFS route_type values
ROUTE_TYPE_RAIL = 2  # Rail
ROUTE_TYPE_BUS = 3   # Bus (to exclude)

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "connexions.csv"
STATIONS_FILE = DATA_DIR / "gares-sncf.csv"


def download_gtfs(url: str, temp_dir: str) -> str:
    """Download GTFS zip file."""
    print(f"Downloading GTFS from {url}...")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    zip_path = os.path.join(temp_dir, "gtfs.zip")

    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return zip_path


def extract_gtfs(zip_path: str, temp_dir: str) -> dict[str, str]:
    """Extract needed GTFS files."""
    print("Extracting GTFS files...")
    needed_files = ["routes.txt", "trips.txt", "stop_times.txt", "stops.txt", "agency.txt"]
    extracted = {}

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            basename = os.path.basename(name)
            if basename in needed_files:
                zf.extract(name, temp_dir)
                extracted[basename] = os.path.join(temp_dir, name)

    return extracted


def normalize_station_name(name: str) -> str:
    """Normalize station name for matching."""
    import unicodedata
    # Remove accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    # Lowercase and remove special chars
    name = name.lower()
    name = name.replace('-', ' ').replace('&', ' ')
    # Remove multiple spaces
    name = ' '.join(name.split())
    return name


def load_station_names_mapping() -> dict[str, str]:
    """Load canonical station names from gares-sncf.csv."""
    if not STATIONS_FILE.exists():
        print(f"Warning: {STATIONS_FILE} not found, skipping name normalization")
        return {}

    mapping = {}
    with open(STATIONS_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical_name = row["nom"].strip()
            normalized = normalize_station_name(canonical_name)
            mapping[normalized] = canonical_name

    print(f"Loaded {len(mapping)} canonical station names")
    return mapping


def find_canonical_name(gtfs_name: str, mapping: dict[str, str]) -> str:
    """Find canonical station name from GTFS name."""
    if not mapping:
        return gtfs_name

    normalized = normalize_station_name(gtfs_name)

    # Exact match
    if normalized in mapping:
        return mapping[normalized]

    # Prefix match (for cases like "Paris Gare de Lyon Hall 1 - 2" -> "Paris - Gare de Lyon - Hall 1 & 2")
    # Try progressively shorter prefixes
    words = normalized.split()
    for i in range(len(words), 2, -1):
        prefix = ' '.join(words[:i])
        for norm_canonical, canonical in mapping.items():
            if norm_canonical.startswith(prefix) or prefix.startswith(norm_canonical):
                # Additional check: both should contain key identifying words
                # For "Gare de Lyon", both should have "gare" and "lyon"
                key_words = [w for w in words if len(w) > 3][:3]
                if all(w in norm_canonical for w in key_words):
                    return canonical

    # No match found
    return gtfs_name


def load_stops(filepath: str, station_mapping: dict[str, str]) -> dict[str, dict]:
    """Load stops from GTFS stops.txt with canonical names."""
    stops = {}
    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stop_id = row["stop_id"]
            gtfs_name = row["stop_name"]
            canonical_name = find_canonical_name(gtfs_name, station_mapping)

            stops[stop_id] = {
                "name": canonical_name,
                "lat": float(row.get("stop_lat", 0) or 0),
                "lon": float(row.get("stop_lon", 0) or 0),
            }
    print(f"Loaded {len(stops)} stops")
    return stops


def load_routes(filepath: str) -> dict[str, dict]:
    """Load rail routes from GTFS routes.txt (exclude buses)."""
    routes = {}
    bus_count = 0

    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            route_type = int(row.get("route_type", 0) or 0)

            # Only include rail (type 2), exclude buses (type 3)
            if route_type == ROUTE_TYPE_BUS:
                bus_count += 1
                continue

            route_id = row["route_id"]

            # Determine train type from route info
            route_name = row.get("route_short_name", "") or row.get("route_long_name", "")
            train_type = classify_train_type(route_name, row.get("agency_id", ""))

            routes[route_id] = {
                "name": route_name,
                "type": train_type,
                "route_type": route_type,
            }

    print(f"Loaded {len(routes)} rail routes (excluded {bus_count} bus routes)")
    return routes


def classify_train_type(route_name: str, agency_id: str) -> str:
    """Classify train type from route name or agency."""
    name_lower = route_name.lower()
    agency_lower = agency_id.lower()

    if "tgv" in name_lower or "inoui" in name_lower:
        return "TGV"
    elif "ter" in name_lower or "ter" in agency_lower:
        return "TER"
    elif "intercit" in name_lower:
        return "Intercites"
    elif "ouigo" in name_lower:
        return "OUIGO"
    elif "transilien" in name_lower or "rer" in name_lower:
        return "Transilien"
    elif "eurostar" in name_lower:
        return "Eurostar"
    elif "thalys" in name_lower or "izy" in name_lower:
        return "Thalys"
    else:
        return "Train"


def load_trips(filepath: str, routes: dict) -> dict[str, str]:
    """Load trips and map to route_id (only for rail routes)."""
    trips = {}  # trip_id -> route_id

    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            route_id = row["route_id"]
            if route_id in routes:  # Only rail routes
                trips[row["trip_id"]] = route_id

    print(f"Loaded {len(trips)} rail trips")
    return trips


def parse_gtfs_time(time_str: str) -> int:
    """Parse GTFS time (HH:MM:SS) to minutes from midnight."""
    if not time_str:
        return 0
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    return hours * 60 + minutes


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two coordinates."""
    import math
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def process_stop_times(filepath: str, trips: dict, routes: dict, stops: dict) -> dict:
    """
    Process stop_times.txt to extract connections with travel times.

    Returns dict: (station1, station2) -> {duration_min, distance, train_types}
    """
    print("Processing stop_times (this may take a while)...")

    # First, group stop_times by trip_id
    trip_stops = defaultdict(list)

    with open(filepath, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trip_id = row["trip_id"]
            if trip_id not in trips:
                continue

            trip_stops[trip_id].append({
                "stop_id": row["stop_id"],
                "arrival": parse_gtfs_time(row.get("arrival_time", "")),
                "departure": parse_gtfs_time(row.get("departure_time", "")),
                "sequence": int(row.get("stop_sequence", 0)),
            })

    print(f"Grouped stops for {len(trip_stops)} trips")

    # Now extract connections between consecutive stops
    connections = defaultdict(lambda: {
        "durations": [],
        "train_types": set(),
    })

    for trip_id, stop_list in tqdm(trip_stops.items(), desc="Extracting connections"):
        route_id = trips[trip_id]
        train_type = routes[route_id]["type"]

        # Sort by sequence
        stop_list.sort(key=lambda x: x["sequence"])

        # Create connections between consecutive stops
        for i in range(len(stop_list) - 1):
            stop1 = stop_list[i]
            stop2 = stop_list[i + 1]

            stop1_id = stop1["stop_id"]
            stop2_id = stop2["stop_id"]

            # Get stop names
            if stop1_id not in stops or stop2_id not in stops:
                continue

            name1 = stops[stop1_id]["name"]
            name2 = stops[stop2_id]["name"]

            # Skip self-connections
            if name1 == name2:
                continue

            # Calculate duration (departure from stop1 to arrival at stop2)
            duration = stop2["arrival"] - stop1["departure"]

            # Handle overnight trips
            if duration < 0:
                duration += 24 * 60

            # Skip unrealistic durations (< 1 min or > 12 hours)
            if duration < 1 or duration > 720:
                continue

            # Use sorted station names as key for bidirectional
            key = tuple(sorted([name1, name2]))
            connections[key]["durations"].append(duration)
            connections[key]["train_types"].add(train_type)

    print(f"Found {len(connections)} unique connections")
    return connections, stops


def save_connections(connections: dict, stops: dict, output_path: Path):
    """Save connections to CSV."""
    print(f"Saving connections to {output_path}...")

    # Build stop name to coords mapping
    name_to_coords = {}
    for stop_id, stop_data in stops.items():
        name = stop_data["name"]
        if name not in name_to_coords:
            name_to_coords[name] = (stop_data["lat"], stop_data["lon"])

    rows = []
    for (station1, station2), data in connections.items():
        # Calculate average/median duration
        durations = data["durations"]
        if not durations:
            continue

        # Use minimum duration (fastest train)
        min_duration = min(durations)

        # Calculate distance
        coords1 = name_to_coords.get(station1)
        coords2 = name_to_coords.get(station2)

        if coords1 and coords2 and coords1[0] and coords2[0]:
            distance = round(haversine(coords1[0], coords1[1], coords2[0], coords2[1]), 2)
        else:
            distance = 0

        # Get dominant train type
        train_types = data["train_types"]
        # Priority: TGV > Intercites > TER > others
        if "TGV" in train_types or "OUIGO" in train_types:
            train_type = "TGV"
        elif "Intercites" in train_types:
            train_type = "Intercites"
        elif "TER" in train_types:
            train_type = "TER"
        elif "Transilien" in train_types:
            train_type = "Transilien"
        else:
            train_type = list(train_types)[0] if train_types else "Train"

        rows.append({
            "station1": station1,
            "station2": station2,
            "distance": distance,
            "duration_min": min_duration,
            "train_type": train_type,
        })

    # Sort by station names
    rows.sort(key=lambda x: (x["station1"], x["station2"]))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["station1", "station2", "distance", "duration_min", "train_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} connections to {output_path}")


def main():
    """Main entry point."""
    # Use consolidated GTFS (TGV + TER + Intercites)
    gtfs_url = GTFS_URLS["consolidated"]

    # Load canonical station names
    print("Loading canonical station names from gares-sncf.csv...")
    station_mapping = load_station_names_mapping()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download and extract
        zip_path = download_gtfs(gtfs_url, temp_dir)
        files = extract_gtfs(zip_path, temp_dir)

        # Check we have all needed files
        required = ["routes.txt", "trips.txt", "stop_times.txt", "stops.txt"]
        for req in required:
            if req not in files:
                print(f"Error: Missing {req} in GTFS archive")
                return

        # Load data with canonical names
        stops = load_stops(files["stops.txt"], station_mapping)
        routes = load_routes(files["routes.txt"])
        trips = load_trips(files["trips.txt"], routes)

        # Process connections
        connections, stops = process_stop_times(files["stop_times.txt"], trips, routes, stops)

        # Save
        save_connections(connections, stops, OUTPUT_FILE)

    print("\nDone!")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
