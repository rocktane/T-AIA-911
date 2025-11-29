"""Resolve city names to train stations."""

import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ..nlp.preprocessing import normalize_city_name
from .nearest_station import NearestStationFinder, Station


@dataclass
class City:
    """City data."""

    name: str
    lat: float
    lon: float
    postal_code: str = ""
    department: str = ""
    region: str = ""


@dataclass
class ResolvedCity:
    """Result of city resolution to station."""

    city_requested: str
    city_matched: str  # The actual city name matched
    station: Station
    distance_km: float
    is_station: bool  # True if city is directly a station
    match_score: float = 1.0  # 1.0 for exact match, < 1.0 for fuzzy


class CityResolver:
    """
    Resolve city names to the nearest train station.

    If the city has a station, returns that station.
    Otherwise, finds the nearest station using GPS coordinates.
    """

    # Minimum similarity score for fuzzy matching (0.0 - 1.0)
    FUZZY_THRESHOLD = 0.85

    def __init__(
        self,
        stations_file: str | Path | None = None,
        cities_file: str | Path | None = None,
    ):
        """
        Initialize resolver with station and city data.

        Args:
            stations_file: Path to stations CSV
            cities_file: Path to cities CSV
        """
        self.station_finder = NearestStationFinder()
        self.cities: dict[str, City] = {}  # Normalized name -> City
        self.city_aliases: dict[str, str] = {}  # Alias (normalized) -> canonical normalized name
        self.station_cities: set[str] = set()  # Cities that have stations

        if stations_file:
            self.load_stations(stations_file)
        if cities_file:
            self.load_cities(cities_file)

    def load_stations(self, filepath: str | Path) -> None:
        """Load station data."""
        self.station_finder.load_stations(filepath)
        # Build set of station city names (normalized)
        self.station_cities = {
            normalize_city_name(s.name) for s in self.station_finder.stations
        }

    def load_cities(self, filepath: str | Path) -> None:
        """
        Load city data from CSV.

        Expected columns: nom_commune, latitude, longitude, code_postal,
                         nom_departement, nom_region
        Also handles nom_commune_postal as alias.
        """
        filepath = Path(filepath)
        self.cities = {}
        self.city_aliases = {}

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Get the canonical name
                    name = row.get("nom_commune", row.get("nom", "")).strip()
                    if not name:
                        continue

                    city = City(
                        name=name,
                        lat=float(row["latitude"]),
                        lon=float(row["longitude"]),
                        postal_code=row.get("code_postal", ""),
                        department=row.get("nom_departement", ""),
                        region=row.get("nom_region", ""),
                    )

                    # Store by normalized name
                    normalized_name = normalize_city_name(name)
                    self.cities[normalized_name] = city

                    # Also index by postal name as alias (e.g., "ST VICTORET")
                    postal_name = row.get("nom_commune_postal", "").strip()
                    if postal_name and postal_name != name:
                        normalized_postal = normalize_city_name(postal_name)
                        if normalized_postal != normalized_name:
                            self.city_aliases[normalized_postal] = normalized_name

                    # Also index complete name as alias
                    complete_name = row.get("nom_commune_complet", "").strip()
                    if complete_name and complete_name != name:
                        normalized_complete = normalize_city_name(complete_name)
                        if normalized_complete != normalized_name:
                            self.city_aliases[normalized_complete] = normalized_name

                except (ValueError, KeyError):
                    continue

    def _find_city_exact(self, normalized_name: str) -> City | None:
        """Find city by exact normalized name match."""
        # Direct lookup
        if normalized_name in self.cities:
            return self.cities[normalized_name]

        # Check aliases
        if normalized_name in self.city_aliases:
            canonical = self.city_aliases[normalized_name]
            return self.cities.get(canonical)

        return None

    def _find_city_fuzzy(self, normalized_name: str) -> tuple[City | None, float]:
        """
        Find city using fuzzy matching.

        Returns:
            Tuple of (City or None, similarity score)
        """
        best_match = None
        best_score = 0.0
        best_city = None

        # Check all cities and aliases
        all_names = list(self.cities.keys()) + list(self.city_aliases.keys())

        for candidate in all_names:
            score = SequenceMatcher(None, normalized_name, candidate).ratio()
            if score > best_score and score >= self.FUZZY_THRESHOLD:
                best_score = score
                best_match = candidate

        if best_match:
            # Get the city (might be via alias)
            if best_match in self.cities:
                best_city = self.cities[best_match]
            elif best_match in self.city_aliases:
                canonical = self.city_aliases[best_match]
                best_city = self.cities.get(canonical)

        return best_city, best_score

    def resolve(self, city_name: str) -> ResolvedCity | None:
        """
        Resolve a city name to a train station.

        Args:
            city_name: Name of the city to resolve

        Returns:
            ResolvedCity with station info, or None if city unknown
        """
        normalized_input = normalize_city_name(city_name)

        # Case 1: City is directly a station (check with normalized name)
        station = self.station_finder.find_by_name(city_name)
        if not station:
            # Try finding station with normalized name
            station = self.station_finder.find_by_normalized_name(normalized_input)

        if station:
            return ResolvedCity(
                city_requested=city_name,
                city_matched=station.name,
                station=station,
                distance_km=0.0,
                is_station=True,
                match_score=1.0,
            )

        # Case 2: Try exact match on city name
        city = self._find_city_exact(normalized_input)
        match_score = 1.0

        # Case 3: Try fuzzy matching if no exact match
        if not city:
            city, match_score = self._find_city_fuzzy(normalized_input)

        if not city:
            return None

        # Find nearest station
        result = self.station_finder.find_nearest(city.lat, city.lon)
        if not result:
            return None

        return ResolvedCity(
            city_requested=city_name,
            city_matched=city.name,
            station=result.station,
            distance_km=result.distance_km,
            is_station=False,
            match_score=match_score,
        )

    def resolve_pair(
        self, departure: str, arrival: str
    ) -> tuple[ResolvedCity | None, ResolvedCity | None]:
        """
        Resolve both departure and arrival cities.

        Args:
            departure: Departure city name
            arrival: Arrival city name

        Returns:
            Tuple of (resolved_departure, resolved_arrival)
        """
        return self.resolve(departure), self.resolve(arrival)

    def get_all_stations_for_city(self, city_name: str) -> list[str]:
        """
        Get all station names for a city (useful for cities with multiple stations).

        Args:
            city_name: Name of the city (e.g., "Paris", "Marseille")

        Returns:
            List of station names matching the city
        """
        stations = self.station_finder.find_all_by_prefix(city_name)
        if stations:
            return [s.name for s in stations]

        # If no prefix match, try exact match
        station = self.station_finder.find_by_name(city_name)
        if station:
            return [station.name]

        # If city exists in communes, find nearest station
        resolved = self.resolve(city_name)
        if resolved:
            return [resolved.station.name]

        return []
