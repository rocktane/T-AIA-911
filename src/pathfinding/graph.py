"""Railway graph construction from SNCF data."""

import csv
from pathlib import Path

import networkx as nx

from src.geo.distance import haversine


class RailwayGraph:
    """
    Graph representation of the French railway network.

    Nodes represent stations, edges represent connections.
    Edge weights are travel time in minutes (preferred) or distance in km.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.graph = nx.Graph()

    def load_stations(self, filepath: str | Path) -> None:
        """
        Load stations as nodes from CSV.

        Expected columns: nom, lon, lat
        """
        filepath = Path(filepath)

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row["nom"].strip()
                    lat = float(row["lat"])
                    lon = float(row["lon"])

                    self.graph.add_node(
                        name,
                        lat=lat,
                        lon=lon,
                        code=row.get("libellecourt", ""),
                    )
                except (ValueError, KeyError):
                    continue

    def load_connections(self, filepath: str | Path) -> None:
        """
        Load connections as edges from CSV.

        Expected columns: station1, station2, distance
        Optional columns: duration_min, train_type

        Uses duration_min as weight if available, otherwise distance.
        """
        filepath = Path(filepath)
        connections_added = 0
        connections_skipped = 0

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    station1 = row["station1"].strip()
                    station2 = row["station2"].strip()
                    distance = float(row.get("distance", 0) or 0)

                    # Duration in minutes (preferred weight)
                    duration = None
                    if "duration_min" in row and row["duration_min"]:
                        duration = float(row["duration_min"])

                    # Train type (TGV, TER, Intercites, etc.)
                    train_type = row.get("train_type", "Train").strip()

                    # Use duration as weight if available, otherwise distance
                    # Duration is better because it reflects actual travel time
                    weight = duration if duration is not None else distance

                    # Skip connections with zero weight
                    if weight <= 0:
                        connections_skipped += 1
                        continue

                    # Only add edge if both stations exist in graph
                    if station1 in self.graph and station2 in self.graph:
                        self.graph.add_edge(
                            station1,
                            station2,
                            weight=weight,
                            distance=distance,
                            duration=duration,
                            train_type=train_type,
                        )
                        connections_added += 1
                    else:
                        # Add stations if they don't exist (from connection data)
                        if station1 not in self.graph:
                            self.graph.add_node(station1)
                        if station2 not in self.graph:
                            self.graph.add_node(station2)
                        self.graph.add_edge(
                            station1,
                            station2,
                            weight=weight,
                            distance=distance,
                            duration=duration,
                            train_type=train_type,
                        )
                        connections_added += 1

                except (ValueError, KeyError):
                    connections_skipped += 1
                    continue

    def build_connections_from_coordinates(
        self, max_distance_km: float = 150.0
    ) -> None:
        """
        Build connections based on geographic proximity.

        This is a fallback when no connection data is available.
        Connects stations within max_distance_km of each other.

        Note: This creates a simplified graph and may not reflect
        actual railway connections.
        """
        stations = list(self.graph.nodes(data=True))

        for i, (name1, data1) in enumerate(stations):
            for name2, data2 in stations[i + 1 :]:
                if "lat" not in data1 or "lat" not in data2:
                    continue
                distance = haversine(
                    data1["lat"], data1["lon"], data2["lat"], data2["lon"]
                )
                if distance <= max_distance_km:
                    # Estimate travel time: ~80 km/h average
                    estimated_duration = (distance / 80) * 60
                    self.graph.add_edge(
                        name1,
                        name2,
                        weight=estimated_duration,
                        distance=distance,
                        duration=estimated_duration,
                        train_type="Train",
                    )

    def get_stations(self) -> list[str]:
        """Get list of all station names."""
        return list(self.graph.nodes())

    def has_station(self, name: str) -> bool:
        """Check if a station exists in the graph."""
        return name in self.graph

    def get_neighbors(self, station: str) -> list[str]:
        """Get neighboring stations."""
        if station not in self.graph:
            return []
        return list(self.graph.neighbors(station))

    def get_edge_weight(self, station1: str, station2: str) -> float | None:
        """Get the weight (duration/distance) between two connected stations."""
        if self.graph.has_edge(station1, station2):
            return self.graph[station1][station2].get("weight")
        return None

    def get_edge_data(self, station1: str, station2: str) -> dict | None:
        """Get all edge data (weight, distance, duration, train_type)."""
        if self.graph.has_edge(station1, station2):
            return dict(self.graph[station1][station2])
        return None

    def get_edge_duration(self, station1: str, station2: str) -> float | None:
        """Get travel duration in minutes between two connected stations."""
        if self.graph.has_edge(station1, station2):
            return self.graph[station1][station2].get("duration")
        return None

    def get_edge_train_type(self, station1: str, station2: str) -> str | None:
        """Get train type for a connection."""
        if self.graph.has_edge(station1, station2):
            return self.graph[station1][station2].get("train_type", "Train")
        return None

    def __len__(self) -> int:
        """Return number of stations."""
        return len(self.graph)
