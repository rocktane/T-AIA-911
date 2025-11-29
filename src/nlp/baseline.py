"""Baseline NER using regex patterns and dictionary matching."""

import re
from dataclasses import dataclass

from .preprocessing import correct_city_typo, find_cities_with_fuzzy, normalize_city_name


@dataclass
class NERResult:
    """Result of NER extraction."""

    departure: str | None
    arrival: str | None
    is_valid: bool
    # Position and confidence information
    departure_start: int | None = None
    departure_end: int | None = None
    arrival_start: int | None = None
    arrival_end: int | None = None
    departure_confidence: float = 0.0
    arrival_confidence: float = 0.0

    def to_tuple(self) -> tuple[str | None, str | None]:
        """Return (departure, arrival) tuple."""
        return (self.departure, self.arrival)


class BaselineNER:
    """
    Baseline NER using regex patterns and dictionary matching.

    This is a simple approach that:
    1. Detects known city names in the text
    2. Uses regex patterns to identify departure/arrival roles
    """

    # Patterns indicating departure (city follows the pattern)
    DEPARTURE_PATTERNS = [
        r"(?:de|depuis|partant de|en partance de|au départ de|au depart de)\s+([A-Za-zÀ-ÿ\s\-']+?)(?:\s+(?:à|a|vers|pour|direction|jusqu|,|\.|\?|$))",
        r"^([A-Za-zÀ-ÿ\s\-']+?)\s*[-–→]\s*[A-Za-zÀ-ÿ]",  # Paris - Lyon
        r"départ\s+(?:de\s+)?([A-Za-zÀ-ÿ\s\-']+?)(?:\s*,|\s+arrivée)",
        r"^train\s+([A-Za-zÀ-ÿ\s\-']+?)\s+[A-Za-zÀ-ÿ]",  # Train Paris Lyon
        r"^billet\s+([A-Za-zÀ-ÿ\s\-']+?)\s+[A-Za-zÀ-ÿ]",  # Billet Paris Lyon
        r"^trajet\s+([A-Za-zÀ-ÿ\s\-']+?)\s*[-–]\s*[A-Za-zÀ-ÿ]",  # Trajet Paris - Lyon
    ]

    # Patterns indicating arrival (city follows the pattern)
    ARRIVAL_PATTERNS = [
        r"(?:à|a|vers|pour|direction|jusqu'à|jusqu'a)\s+([A-Za-zÀ-ÿ\s\-']+?)(?:\s+(?:depuis|de|en partant|,|\.|\?|$))",
        r"[A-Za-zÀ-ÿ]\s*[-–→]\s*([A-Za-zÀ-ÿ\s\-']+?)(?:\s*$|\s*\?|\s*\.)",  # Paris - Lyon
        r"arrivée\s+(?:à\s+)?([A-Za-zÀ-ÿ\s\-']+?)(?:\s*$|\s*\.)",
    ]

    def __init__(self, cities: list[str], use_fuzzy: bool = True, fuzzy_threshold: int = 85):
        """
        Initialize with a list of known cities.

        Args:
            cities: List of city names
            use_fuzzy: Enable fuzzy matching for typo correction
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.cities = set(cities)
        self.cities_list = list(cities)  # Keep list for fuzzy matching
        self.cities_normalized = {normalize_city_name(c): c for c in cities}
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

        # Compile patterns
        self.dep_patterns = [re.compile(p, re.IGNORECASE) for p in self.DEPARTURE_PATTERNS]
        self.arr_patterns = [re.compile(p, re.IGNORECASE) for p in self.ARRIVAL_PATTERNS]

    def _find_city_in_text(self, text: str) -> list[tuple[str, int, int]]:
        """
        Find all city names in text with their positions.

        Returns:
            List of (city_name, start_pos, end_pos)
        """
        text_lower = text.lower()
        found = []

        for city in self.cities:
            city_lower = city.lower()
            start = 0
            while True:
                pos = text_lower.find(city_lower, start)
                if pos == -1:
                    break
                # Check word boundaries
                before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
                after_ok = (
                    pos + len(city_lower) >= len(text_lower)
                    or not text_lower[pos + len(city_lower)].isalnum()
                )
                if before_ok and after_ok:
                    found.append((city, pos, pos + len(city_lower)))
                start = pos + 1

        # Sort by position
        found.sort(key=lambda x: x[1])
        return found

    def _match_pattern(self, text: str, patterns: list[re.Pattern]) -> str | None:
        """Try to match patterns and return the captured city if valid."""
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                candidate = match.group(1).strip()
                # Check if candidate is a known city (exact match)
                candidate_norm = normalize_city_name(candidate)
                if candidate_norm in self.cities_normalized:
                    return self.cities_normalized[candidate_norm]
                # Try fuzzy matching for typos
                if self.use_fuzzy:
                    corrected = correct_city_typo(
                        candidate, self.cities_list, threshold=self.fuzzy_threshold
                    )
                    if corrected:
                        return corrected
        return None

    def _find_cities_fuzzy(self, text: str) -> list[tuple[str, str, int]]:
        """
        Find cities in text using fuzzy matching.

        Returns:
            List of (original_text, matched_city, score)
        """
        if not self.use_fuzzy:
            return []
        return find_cities_with_fuzzy(text, self.cities_list, threshold=self.fuzzy_threshold)

    def extract(self, sentence: str) -> NERResult:
        """
        Extract departure and arrival from a sentence.

        Args:
            sentence: Input sentence

        Returns:
            NERResult with departure, arrival, and validity
        """
        # Find all cities in the text (exact match)
        cities_found = self._find_city_in_text(sentence)

        # Try pattern matching first (includes fuzzy correction)
        departure = self._match_pattern(sentence, self.dep_patterns)
        arrival = self._match_pattern(sentence, self.arr_patterns)

        # If pattern matching found both, validate and return
        if departure and arrival and departure != arrival:
            return NERResult(departure=departure, arrival=arrival, is_valid=True)

        # If we have enough exact matches, use position-based heuristic
        if len(cities_found) >= 2:
            dep_candidate = cities_found[0][0]
            arr_candidate = cities_found[1][0]

            # Check for "à X depuis Y" pattern (arrival before departure)
            arr_first_pattern = re.search(
                r"(?:à|a|vers)\s+\w+.*(?:depuis|de|partant)",
                sentence,
                re.IGNORECASE,
            )
            if arr_first_pattern:
                dep_candidate, arr_candidate = arr_candidate, dep_candidate

            return NERResult(
                departure=dep_candidate, arrival=arr_candidate, is_valid=True
            )

        # Fallback: try fuzzy matching for typos
        if self.use_fuzzy and len(cities_found) < 2:
            fuzzy_matches = self._find_cities_fuzzy(sentence)

            # Combine exact and fuzzy matches
            all_cities = [(c, pos, pos) for c, pos, _ in cities_found]
            for orig_text, matched_city, score in fuzzy_matches:
                # Find position of original text in sentence
                pos = sentence.lower().find(orig_text.lower())
                if pos != -1 and matched_city not in [c[0] for c in all_cities]:
                    all_cities.append((matched_city, pos, pos + len(orig_text)))

            # Sort by position
            all_cities.sort(key=lambda x: x[1])

            if len(all_cities) >= 2:
                dep_candidate = all_cities[0][0]
                arr_candidate = all_cities[1][0]

                # Check for "à X depuis Y" pattern (arrival before departure)
                arr_first_pattern = re.search(
                    r"(?:à|a|vers)\s+\w+.*(?:depuis|de|partant)",
                    sentence,
                    re.IGNORECASE,
                )
                if arr_first_pattern:
                    dep_candidate, arr_candidate = arr_candidate, dep_candidate

                return NERResult(
                    departure=dep_candidate, arrival=arr_candidate, is_valid=True
                )

        return NERResult(departure=None, arrival=None, is_valid=False)

    def process_batch(self, sentences: list[str]) -> list[NERResult]:
        """Process multiple sentences."""
        return [self.extract(s) for s in sentences]
