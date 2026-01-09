"""Baseline NER using regex patterns and dictionary matching."""

import re
from dataclasses import dataclass, field

from .preprocessing import correct_city_typo, find_cities_with_fuzzy, normalize_city_name


@dataclass
class ViaPoint:
    """A single VIA waypoint."""

    city: str
    start: int | None = None
    end: int | None = None
    confidence: float = 0.0
    order: int = 0  # Position in the VIA sequence


@dataclass
class NERResult:
    """Result of NER extraction."""

    departure: str | None
    arrival: str | None
    is_valid: bool
    # VIA waypoints (intermediate stops)
    vias: list[ViaPoint] = field(default_factory=list)
    # Position and confidence information
    departure_start: int | None = None
    departure_end: int | None = None
    arrival_start: int | None = None
    arrival_end: int | None = None
    departure_confidence: float = 0.0
    arrival_confidence: float = 0.0

    def to_tuple(self) -> tuple[str | None, str | None]:
        """Return (departure, arrival) tuple for backward compatibility."""
        return (self.departure, self.arrival)

    def get_via_cities(self) -> list[str]:
        """Return list of VIA city names in order."""
        return [v.city for v in sorted(self.vias, key=lambda x: x.order)]

    def get_full_route(self) -> list[str]:
        """Return complete ordered route: [departure, via1, via2, ..., arrival]."""
        route = []
        if self.departure:
            route.append(self.departure)
        route.extend(self.get_via_cities())
        if self.arrival:
            route.append(self.arrival)
        return route


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

    # Patterns indicating VIA (intermediate waypoints)
    # Note: Use (?=\s|$|[,\.\?]) lookahead for end boundaries without consuming
    VIA_PATTERNS = [
        r"(?:via|par)\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:et|puis|à|a|vers|jusqu)|[,\.\?]|$)",
        r"(?:en passant par|passant par)\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:et|puis|à|a|vers|jusqu)|[,\.\?]|$)",
        r"(?:avec (?:un )?arr[êe]t [àa]|avec escale [àa])\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:et|puis)|[,\.\?]|$)",
        r"(?:avec (?:une )?correspondance [àa]|avec changement [àa])\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:et|puis)|[,\.\?]|$)",
        r"(?:en faisant [eé]tape [àa])\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:et|puis)|[,\.\?]|$)",
        # "puis X puis Y" pattern - captures intermediate cities
        r"(?:puis)\s+([A-Za-zÀ-ÿ\-']+(?:\s+[A-Za-zÀ-ÿ\-']+)*)(?=\s+(?:puis|vers|à|a)|[,\.\?]|$)",
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
        self.via_patterns = [re.compile(p, re.IGNORECASE) for p in self.VIA_PATTERNS]

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

    def _extract_vias(
        self, sentence: str, departure: str | None, arrival: str | None
    ) -> list[ViaPoint]:
        """
        Extract VIA waypoints from sentence, excluding departure and arrival.

        Args:
            sentence: Input sentence
            departure: Departure city (to exclude)
            arrival: Arrival city (to exclude)

        Returns:
            List of ViaPoint objects ordered by position in text
        """
        vias = []
        excluded = {
            departure.lower() if departure else "",
            arrival.lower() if arrival else "",
        }
        seen_cities = set()

        for pattern in self.via_patterns:
            for match in pattern.finditer(sentence):
                candidate = match.group(1).strip()
                candidate_norm = normalize_city_name(candidate)

                # Try exact match first
                matched_city = None
                if candidate_norm in self.cities_normalized:
                    matched_city = self.cities_normalized[candidate_norm]
                elif self.use_fuzzy:
                    # Try fuzzy matching
                    corrected = correct_city_typo(
                        candidate, self.cities_list, threshold=self.fuzzy_threshold
                    )
                    if corrected:
                        matched_city = corrected

                if matched_city:
                    city_lower = matched_city.lower()
                    # Skip if it's departure or arrival
                    if city_lower in excluded:
                        continue
                    # Skip if already found
                    if city_lower in seen_cities:
                        continue

                    seen_cities.add(city_lower)
                    vias.append(
                        ViaPoint(
                            city=matched_city,
                            start=match.start(1),
                            end=match.end(1),
                            confidence=0.8,
                            order=match.start(1),
                        )
                    )

        # Sort by position and assign proper order indices
        vias.sort(key=lambda v: v.start if v.start else 0)
        for i, v in enumerate(vias):
            v.order = i

        return vias

    def extract(self, sentence: str) -> NERResult:
        """
        Extract departure, arrival, and VIA waypoints from a sentence.

        Args:
            sentence: Input sentence

        Returns:
            NERResult with departure, arrival, vias, and validity
        """
        # Find all cities in the text (exact match)
        cities_found = self._find_city_in_text(sentence)

        # Try pattern matching first (includes fuzzy correction)
        departure = self._match_pattern(sentence, self.dep_patterns)
        arrival = self._match_pattern(sentence, self.arr_patterns)

        # If pattern matching found both, validate and return
        if departure and arrival and departure != arrival:
            vias = self._extract_vias(sentence, departure, arrival)
            return NERResult(
                departure=departure, arrival=arrival, vias=vias, is_valid=True
            )

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

            vias = self._extract_vias(sentence, dep_candidate, arr_candidate)
            return NERResult(
                departure=dep_candidate, arrival=arr_candidate, vias=vias, is_valid=True
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

                vias = self._extract_vias(sentence, dep_candidate, arr_candidate)
                return NERResult(
                    departure=dep_candidate, arrival=arr_candidate, vias=vias, is_valid=True
                )

        return NERResult(departure=None, arrival=None, is_valid=False)

    def process_batch(self, sentences: list[str]) -> list[NERResult]:
        """Process multiple sentences."""
        return [self.extract(s) for s in sentences]
