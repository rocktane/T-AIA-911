"""Find nearest train station using KD-Tree for efficient lookup."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..nlp.preprocessing import normalize_city_name
from .distance import haversine


@dataclass
class Station:
    """Train station data."""

    name: str
    code: str
    lat: float
    lon: float
    insee: str


@dataclass
class NearestResult:
    """Result of nearest station search."""

    station: Station
    distance_km: float


class NearestStationFinder:
    """
    Find nearest train stations using KD-Tree for O(log n) lookup.

    The KD-Tree is built on coordinates converted to radians for better
    distance approximation at the Earth's surface.
    """

    def __init__(self, stations_file: str | Path | None = None):
        """
        Initialize finder with station data.

        Args:
            stations_file: Path to CSV file with station data
        """
        self.stations: list[Station] = []
        self.stations_by_normalized: dict[str, Station] = {}  # Normalized name -> Station
        self.tree: cKDTree | None = None
        self.R = 6371  # Earth's radius in km

        if stations_file:
            self.load_stations(stations_file)

    def load_stations(self, filepath: str | Path) -> None:
        """
        Load stations from CSV file.

        Expected columns: nom, libellecourt, lon, lat, codeinsee
        """
        filepath = Path(filepath)
        self.stations = []
        self.stations_by_normalized = {}

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    station = Station(
                        name=row["nom"].strip(),
                        code=row.get("libellecourt", "").strip(),
                        lat=float(row["lat"]),
                        lon=float(row["lon"]),
                        insee=row.get("codeinsee", "").strip(),
                    )
                    self.stations.append(station)
                    # Index by normalized name for better matching
                    normalized = normalize_city_name(station.name)
                    self.stations_by_normalized[normalized] = station
                except (ValueError, KeyError):
                    continue

        self._build_tree()

    def _build_tree(self) -> None:
        """Build KD-Tree from station coordinates."""
        if not self.stations:
            self.tree = None
            return

        # Convert to radians for better distance approximation
        coords = np.array([[s.lat, s.lon] for s in self.stations])
        coords_rad = np.radians(coords)
        self.tree = cKDTree(coords_rad)

    def find_nearest(
        self, lat: float, lon: float, k: int = 1
    ) -> list[NearestResult] | NearestResult | None:
        """
        Find the k nearest stations to a given point.

        Args:
            lat: Latitude of the query point (degrees)
            lon: Longitude of the query point (degrees)
            k: Number of nearest stations to return

        Returns:
            NearestResult or list of NearestResult, or None if no stations
        """
        if self.tree is None or not self.stations:
            return None

        # Convert query point to radians
        point_rad = np.radians([lat, lon])

        # Query KD-Tree
        distances_rad, indices = self.tree.query(point_rad, k=k)

        # Handle single result vs multiple
        if k == 1:
            idx = int(indices)
            station = self.stations[idx]
            # Calculate actual distance using Haversine
            distance = haversine(lat, lon, station.lat, station.lon)
            return NearestResult(station=station, distance_km=round(distance, 1))

        # Multiple results
        results = []
        for dist_rad, idx in zip(
            np.atleast_1d(distances_rad), np.atleast_1d(indices)
        ):
            idx = int(idx)
            station = self.stations[idx]
            distance = haversine(lat, lon, station.lat, station.lon)
            results.append(NearestResult(station=station, distance_km=round(distance, 1)))

        return results

    def find_by_name(self, name: str) -> Station | None:
        """
        Find a station by exact name match (case-insensitive).

        Args:
            name: Station name to search for

        Returns:
            Station or None if not found
        """
        name_lower = name.lower()
        for station in self.stations:
            if station.name.lower() == name_lower:
                return station
        return None

    def find_by_normalized_name(self, normalized_name: str) -> Station | None:
        """
        Find a station by normalized name.

        The normalized name should already be processed with normalize_city_name().

        Args:
            normalized_name: Normalized station name to search for

        Returns:
            Station or None if not found
        """
        return self.stations_by_normalized.get(normalized_name)

    def get_station_names(self) -> list[str]:
        """Get list of all station names."""
        return [s.name for s in self.stations]

    def find_all_by_prefix(self, prefix: str) -> list[Station]:
        """
        Find all stations whose name starts with the given prefix.

        Args:
            prefix: Station name prefix to search for (case-insensitive)

        Returns:
            List of matching stations
        """
        prefix_lower = prefix.lower()
        matches = []
        for station in self.stations:
            station_lower = station.name.lower()
            # Match exact or prefix followed by space
            if station_lower == prefix_lower:
                matches.append(station)
            elif station_lower.startswith(prefix_lower + " "):
                matches.append(station)
            # For hyphen, only match if it's a simple suffix (not a compound name)
            # e.g., "Paris-Nord" OK, but "Marseille-en-Beauvaisis" NOT OK
            elif station_lower.startswith(prefix_lower + "-"):
                # Check if what follows is just one word (no further hyphens or spaces)
                suffix = station_lower[len(prefix_lower) + 1:]
                if " " not in suffix and "-" not in suffix:
                    matches.append(station)
        return matches
