"""
Travel Order Resolver - Main entry point.

Usage:
    cat input.csv | python -m src.main
    python -m src.main input.csv
    python -m src.main --help
"""

import argparse
import csv
import sys
from pathlib import Path

from src.geo import CityResolver
from src.nlp import BaselineNER, SpacyNER
from src.pathfinding import PathFinder, RailwayGraph

# Default data paths
DATA_DIR = Path(__file__).parent.parent / "data"
STATIONS_FILE = DATA_DIR / "gares-sncf.csv"
CITIES_FILE = DATA_DIR / "communes-france.csv"
CONNECTIONS_FILE = DATA_DIR / "connexions.csv"


def load_city_names(filepath: Path) -> list[str]:
    """Load city names from stations CSV."""
    names = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("nom", "").strip()
            if name:
                names.append(name)
    return names


def extract_city_prefix(station_name: str) -> str:
    """
    Extract the city name prefix from a station name.

    Examples:
        "Paris Est" -> "Paris"
        "Marseille Saint-Charles" -> "Marseille"
        "Paris - Gare de Lyon - Hall 1 & 2" -> "Paris"
    """
    if " - " in station_name:
        station_name = station_name.split(" - ")[0]
    parts = station_name.split()
    return parts[0] if parts else station_name


def process_sentence(
    sentence_id: str,
    sentence: str,
    ner,
    resolver: CityResolver,
    pathfinder: PathFinder | None,
    full_output: bool = True,
) -> str:
    """
    Process a single sentence and return formatted output.

    Args:
        sentence_id: ID of the sentence
        sentence: The sentence to process
        ner: NER model (BaselineNER or SpacyNER)
        resolver: CityResolver for finding stations
        pathfinder: PathFinder for routes (optional)
        full_output: Include full details (stations, route)

    Returns:
        Formatted output string
    """
    # Step 1: NLP extraction
    result = ner.extract(sentence)

    if not result.is_valid:
        return f"{sentence_id},INVALID"

    departure, arrival = result.departure, result.arrival
    via_cities = result.get_via_cities()  # List of VIA city names

    if not full_output:
        via_str = "|".join(via_cities) if via_cities else ""
        return f"{sentence_id},{departure},{arrival},{via_str}"

    # Step 2: Extract city prefixes and find all possible stations
    dep_prefix = extract_city_prefix(departure) if departure else None
    arr_prefix = extract_city_prefix(arrival) if arrival else None

    dep_stations = resolver.get_all_stations_for_city(dep_prefix) if dep_prefix else []
    arr_stations = resolver.get_all_stations_for_city(arr_prefix) if arr_prefix else []

    if not dep_stations or not arr_stations:
        return f"{sentence_id},{dep_prefix},{arr_prefix},UNKNOWN_CITY"

    # Step 3: Resolve VIA cities to stations
    via_prefixes = []
    via_stations_list = []
    for via_city in via_cities:
        via_prefix = extract_city_prefix(via_city)
        via_prefixes.append(via_prefix)
        via_stations = resolver.get_all_stations_for_city(via_prefix) if via_prefix else []
        via_stations_list.append(via_stations)

    # Step 4: Find best route among all station combinations
    dep_station = dep_stations[0]
    arr_station = arr_stations[0]
    via_station_names = []

    if pathfinder:
        if via_stations_list and all(vs for vs in via_stations_list):
            # Use waypoint-aware pathfinding
            path_result = pathfinder.find_best_path_multi_with_waypoints(
                dep_stations, arr_stations, via_stations_list
            )
        else:
            path_result = pathfinder.find_best_path_multi(dep_stations, arr_stations)

        if path_result.found:
            route = "→".join(path_result.path)
            dep_station = path_result.path[0]
            arr_station = path_result.path[-1]
            # Extract VIA station names from the path (stations between first and last)
            if len(path_result.path) > 2:
                via_station_names = path_result.path[1:-1]
        else:
            route = "NO_PATH"
    else:
        # Simple route without pathfinding
        via_simple = [vs[0] for vs in via_stations_list if vs]
        all_stops = [dep_station] + via_simple + [arr_station]
        route = "→".join(all_stops)
        via_station_names = via_simple

    # Format output with VIA columns
    via_cities_str = "|".join(via_prefixes) if via_prefixes else ""
    via_stations_str = "|".join(via_station_names) if via_station_names else ""

    return f'{sentence_id},{dep_prefix},{dep_station},{arr_prefix},{arr_station},{via_cities_str},{via_stations_str},"{route}"'


def main():
    parser = argparse.ArgumentParser(
        description="Travel Order Resolver - Extract travel info from French sentences"
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file (default: stdin)",
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "spacy"],
        default="spacy",
        help="NER model to use (default: spacy)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Simple output (only departure/arrival, no stations/routes)",
    )
    parser.add_argument(
        "--stations",
        type=Path,
        default=STATIONS_FILE,
        help="Path to stations CSV",
    )
    parser.add_argument(
        "--cities",
        type=Path,
        default=CITIES_FILE,
        help="Path to cities CSV",
    )
    parser.add_argument(
        "--connections",
        type=Path,
        default=CONNECTIONS_FILE,
        help="Path to connections CSV",
    )

    args = parser.parse_args()

    # Load city names for NER
    if not args.stations.exists():
        print(f"Error: Stations file not found: {args.stations}", file=sys.stderr)
        sys.exit(1)

    cities = load_city_names(args.stations)

    # Initialize NER
    if args.model == "baseline":
        ner = BaselineNER(cities)
    else:
        ner = SpacyNER(cities)

    # Initialize resolver
    resolver = CityResolver(args.stations, args.cities if args.cities.exists() else None)

    # Initialize pathfinder
    pathfinder = None
    if not args.simple:
        graph = RailwayGraph()
        graph.load_stations(args.stations)
        if args.connections.exists():
            graph.load_connections(args.connections)
        else:
            # Build approximate connections from coordinates
            graph.build_connections_from_coordinates(max_distance_km=150)
        pathfinder = PathFinder(graph)

    # Read input
    if args.input:
        input_file = open(args.input, encoding="utf-8")
    else:
        input_file = sys.stdin

    # Process each line
    reader = csv.reader(input_file)
    for row in reader:
        if len(row) < 2:
            continue

        sentence_id = row[0].strip()
        sentence = row[1].strip()

        # Skip header
        if sentence_id.lower() == "sentenceid":
            continue

        output = process_sentence(
            sentence_id,
            sentence,
            ner,
            resolver,
            pathfinder,
            full_output=not args.simple,
        )
        print(output)

    if args.input:
        input_file.close()


if __name__ == "__main__":
    main()
