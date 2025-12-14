"""spaCy-based NER for extracting departure and arrival cities."""

import re
from pathlib import Path

import spacy
from spacy.language import Language

from .baseline import NERResult, ViaPoint
from .preprocessing import correct_city_typo, normalize_city_name


class SpacyNER:
    """
    NER using spaCy for entity detection.

    Can use either:
    - Pre-trained French model (fr_core_news_md) for LOC detection
    - Custom trained model for DEPART/ARRIVEE labels
    """

    # Major cities mapping: simple city name -> main station prefix
    # These are prioritized over fuzzy matching
    MAJOR_CITIES = {
        "paris": "Paris",
        "lyon": "Lyon",
        "marseille": "Marseille",
        "toulouse": "Toulouse",
        "bordeaux": "Bordeaux",
        "lille": "Lille",
        "nantes": "Nantes",
        "strasbourg": "Strasbourg",
        "nice": "Nice",
        "montpellier": "Montpellier",
        "rennes": "Rennes",
        "grenoble": "Grenoble",
        "rouen": "Rouen",
        "toulon": "Toulon",
        "dijon": "Dijon",
        "angers": "Angers",
        "nimes": "Nîmes",
        "orleans": "Orléans",
        "clermont ferrand": "Clermont-Ferrand",
        "tours": "Tours",
        "reims": "Reims",
        "le havre": "Le Havre",
        "le mans": "Le Mans",
        "aix en provence": "Aix-en-Provence",
        "brest": "Brest",
        "limoges": "Limoges",
        "amiens": "Amiens",
        "perpignan": "Perpignan",
        "besancon": "Besançon",
        "metz": "Metz",
        "avignon": "Avignon",
        "nancy": "Nancy",
        "caen": "Caen",
        "saint etienne": "Saint-Étienne",
        "mulhouse": "Mulhouse",
        "poitiers": "Poitiers",
        "dunkerque": "Dunkerque",
        "calais": "Calais",
        "valence": "Valence",
        "chambery": "Chambéry",
        "troyes": "Troyes",
        "la rochelle": "La Rochelle",
        "colmar": "Colmar",
        "quimper": "Quimper",
        "lorient": "Lorient",
        "vannes": "Vannes",
        "saint malo": "Saint-Malo",
        "saint brieuc": "Saint-Brieuc",
        "angouleme": "Angoulême",
        "pau": "Pau",
        "bayonne": "Bayonne",
        "biarritz": "Biarritz",
        "tarbes": "Tarbes",
        "beziers": "Béziers",
    }

    def __init__(
        self,
        cities: list[str],
        communes: list[str] | None = None,
        model_path: str | Path | None = None,
        use_pretrained: bool = True,
        use_fuzzy: bool = True,
        fuzzy_threshold: int = 85,
    ):
        """
        Initialize spaCy NER.

        Args:
            cities: List of known city names for validation (stations)
            communes: List of commune names for extended matching
            model_path: Path to custom trained model (optional)
            use_pretrained: Use pretrained French model if no custom model
            use_fuzzy: Enable fuzzy matching for typo correction
            fuzzy_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.cities = set(cities)
        self.cities_list = list(cities)  # Keep list for fuzzy matching
        self.cities_normalized = {normalize_city_name(c): c for c in cities}
        self.use_fuzzy = use_fuzzy
        self.fuzzy_threshold = fuzzy_threshold

        # Build major cities to main station mapping
        self._build_major_cities_mapping()

        # Build set of major city normalized names for quick lookup
        self.major_city_names = set(normalize_city_name(k) for k in self.MAJOR_CITIES.keys())

        # Also index communes for better coverage
        self.communes = set(communes) if communes else set()
        self.communes_normalized = {normalize_city_name(c): c for c in self.communes}

        # Combined set for dictionary matching (prefer cities/stations)
        self.all_places = list(self.cities) + [c for c in self.communes if c not in self.cities]

        # Load model
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
            self.custom_model = True
        elif use_pretrained:
            try:
                self.nlp = spacy.load("fr_core_news_md")
            except OSError:
                # Fallback to small model
                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                except OSError:
                    raise RuntimeError(
                        "No French spaCy model found. Install with: "
                        "python -m spacy download fr_core_news_md"
                    )
            self.custom_model = False
        else:
            self.nlp = spacy.blank("fr")
            self.custom_model = False

    # Main station keywords - stations containing these are preferred
    MAIN_STATION_KEYWORDS = [
        "saint charles", "saint-charles",  # Marseille
        "part dieu", "part-dieu",  # Lyon
        "montparnasse",  # Paris
        "gare de lyon",  # Paris
        "saint jean", "saint-jean",  # Bordeaux
        "matabiau",  # Toulouse
        "flandres", "europe",  # Lille
        "ville",  # Generic main station suffix
        "central", "centre",
    ]

    def _build_major_cities_mapping(self):
        """Build mapping from major city names to their main stations.

        This ensures that when we detect 'Marseille', we return 'Marseille Saint-Charles'
        instead of 'Marseille-en-Beauvaisis'.
        """
        self.major_cities_to_station = {}

        for city_key in self.MAJOR_CITIES.keys():
            city_norm = normalize_city_name(city_key)

            # Find all stations that match this city name
            matching_stations = [
                station
                for station_norm, station in self.cities_normalized.items()
                if station_norm.startswith(city_norm + " ") or station_norm == city_norm
            ]

            if matching_stations:
                # First, try to find a main station by keywords
                main_station = None
                for station in matching_stations:
                    station_lower = station.lower()
                    for keyword in self.MAIN_STATION_KEYWORDS:
                        if keyword in station_lower:
                            main_station = station
                            break
                    if main_station:
                        break

                # If no main station found by keyword, take the shortest (fallback)
                if not main_station:
                    main_station = min(matching_stations, key=len)

                self.major_cities_to_station[city_norm] = main_station

    def _clean_entity_text(self, text: str) -> str:
        """Clean entity text by removing leading/trailing prepositions and common words."""
        text = text.strip()

        # Remove common French prepositions at the start
        leading_words = ["de ", "d'", "à ", "a ", "vers ", "pour ", "depuis ", "la ", "le ", "les ", "l'"]
        text_lower = text.lower()
        for prep in leading_words:
            if text_lower.startswith(prep):
                text = text[len(prep):]
                text_lower = text.lower()
                break

        # Remove trailing noise - keep removing until clean
        # These patterns can happen with NER detection errors (e.g., "Marseille en partant")
        trailing_patterns = [
            " en partant de", " en partant", " partant de", " partant",
            " en arrivant", " arrivant",
            " depuis", " vers", " pour", " direction",
            " de", " à", " a", " en",
        ]

        changed = True
        while changed:
            changed = False
            text_lower = text.lower()
            for pattern in trailing_patterns:
                if text_lower.endswith(pattern):
                    text = text[: -len(pattern)]
                    changed = True
                    break

        return text.strip()

    def _validate_city(self, text: str) -> str | None:
        """Check if text is a known city and return canonical name.

        Priority order:
        1. Major cities (exact match) -> return main station
        2. Exact match in stations
        3. Prefix match (only for non-major cities to avoid false matches)
        """
        # Clean entity text first (remove prepositions like "de", "à")
        text = self._clean_entity_text(text)
        text_norm = normalize_city_name(text)

        # Priority 1: Check if this is a major city (exact match only)
        # This prevents "Marseille" from matching "Marseille-en-Beauvaisis"
        if text_norm in self.major_cities_to_station:
            return self.major_cities_to_station[text_norm]

        # Priority 2: Exact match in stations
        if text_norm in self.cities_normalized:
            return self.cities_normalized[text_norm]

        # Priority 3: Prefix match for non-major cities
        # Skip prefix matching if text is a major city name (already handled above)
        if text_norm not in self.major_city_names:
            matches = []
            for city_norm, city_original in self.cities_normalized.items():
                # Check if city starts with the search term (followed by space or end)
                if city_norm.startswith(text_norm + " ") or city_norm == text_norm:
                    matches.append(city_original)

            if matches:
                # Return the shortest match (most likely the main station)
                return min(matches, key=len)

        # Priority 4: Fuzzy matching for typo correction
        if self.use_fuzzy and len(text) >= 4:  # Only fuzzy match for longer names
            corrected = correct_city_typo(
                text, self.cities_list, threshold=self.fuzzy_threshold
            )
            if corrected:
                return corrected

        return None

    def _split_compound_entity(self, text: str) -> tuple[str | None, str | None]:
        """Try to split an entity that contains multiple cities.

        Handles patterns like "Paris - Marseille", "Lyon / Bordeaux", "Nantes > Rennes".
        Returns (departure, arrival) or (None, None) if not splittable.
        """
        separators = [" - ", " – ", " — ", " / ", " > ", " → "]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) == 2:
                    dep = self._validate_city(parts[0].strip())
                    arr = self._validate_city(parts[1].strip())
                    if dep and arr:
                        return dep, arr
        return None, None

    def _extract_with_custom_model(self, doc) -> NERResult:
        """Extract using custom DEPART/ARRIVEE/VIA labels."""
        departure = None
        arrival = None
        vias = []
        dep_start, dep_end = None, None
        arr_start, arr_end = None, None
        dep_confidence, arr_confidence = 0.0, 0.0

        for ent in doc.ents:
            if ent.label_ == "DEPART":
                validated = self._validate_city(ent.text)
                if validated:
                    departure = validated
                    dep_start = ent.start_char
                    dep_end = ent.end_char
                    # spaCy doesn't provide confidence for NER, use 1.0 as default
                    dep_confidence = 1.0
                elif not departure and not arrival:
                    # Try to split compound entity (e.g., "Paris - Marseille")
                    dep, arr = self._split_compound_entity(ent.text)
                    if dep and arr:
                        departure = dep
                        arrival = arr
                        dep_start = ent.start_char
                        dep_end = ent.end_char
                        dep_confidence = 0.9
                        arr_confidence = 0.9
            elif ent.label_ == "ARRIVEE":
                validated = self._validate_city(ent.text)
                if validated:
                    arrival = validated
                    arr_start = ent.start_char
                    arr_end = ent.end_char
                    arr_confidence = 1.0
                elif not departure and not arrival:
                    # Try to split compound entity
                    dep, arr = self._split_compound_entity(ent.text)
                    if dep and arr:
                        departure = dep
                        arrival = arr
                        arr_start = ent.start_char
                        arr_end = ent.end_char
                        dep_confidence = 0.9
                        arr_confidence = 0.9
            elif ent.label_ == "VIA":
                validated = self._validate_city(ent.text)
                if validated:
                    vias.append(ViaPoint(
                        city=validated,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=1.0,
                        order=ent.start_char,
                    ))

        # Fallback for simple patterns like "Paris - Marseille" or "Paris Marseille"
        # when no DEPART/ARRIVEE labels are detected
        if departure is None or arrival is None:
            text = doc.text
            # Try to find cities by dictionary matching
            locations = self._find_cities_in_text(text, [])

            if len(locations) >= 2:
                # Sort by position
                locations.sort(key=lambda x: x["start"])

                # Use context-based classification first
                dep_ctx, arr_ctx, dep_loc, arr_loc = self._classify_locations_with_info(text, locations)

                if dep_ctx and arr_ctx and dep_ctx != arr_ctx:
                    # Both found via context
                    departure = dep_ctx
                    arrival = arr_ctx
                    dep_start = dep_loc["start"] if dep_loc else None
                    dep_end = dep_loc["end"] if dep_loc else None
                    arr_start = arr_loc["start"] if arr_loc else None
                    arr_end = arr_loc["end"] if arr_loc else None
                    dep_confidence = dep_loc.get("confidence", 0.7) if dep_loc else 0.7
                    arr_confidence = arr_loc.get("confidence", 0.7) if arr_loc else 0.7
                elif dep_ctx and not arr_ctx:
                    # Only departure found via context, other location is arrival
                    departure = dep_ctx
                    dep_start = dep_loc["start"] if dep_loc else None
                    dep_end = dep_loc["end"] if dep_loc else None
                    dep_confidence = dep_loc.get("confidence", 0.7) if dep_loc else 0.7
                    # Find the other location for arrival
                    for loc in locations:
                        if loc["text"] != dep_ctx:
                            arrival = loc["text"]
                            arr_start = loc["start"]
                            arr_end = loc["end"]
                            arr_confidence = loc.get("confidence", 0.6)
                            break
                elif arr_ctx and not dep_ctx:
                    # Only arrival found via context, other location is departure
                    arrival = arr_ctx
                    arr_start = arr_loc["start"] if arr_loc else None
                    arr_end = arr_loc["end"] if arr_loc else None
                    arr_confidence = arr_loc.get("confidence", 0.7) if arr_loc else 0.7
                    # Find the other location for departure
                    for loc in locations:
                        if loc["text"] != arr_ctx:
                            departure = loc["text"]
                            dep_start = loc["start"]
                            dep_end = loc["end"]
                            dep_confidence = loc.get("confidence", 0.6)
                            break
                else:
                    # No context found - simple fallback: first city = departure, second = arrival
                    departure = locations[0]["text"]
                    arrival = locations[1]["text"]
                    dep_start = locations[0]["start"]
                    dep_end = locations[0]["end"]
                    arr_start = locations[1]["start"]
                    arr_end = locations[1]["end"]
                    dep_confidence = locations[0].get("confidence", 0.6)
                    arr_confidence = locations[1].get("confidence", 0.6)

        is_valid = departure is not None and arrival is not None

        # Sort VIAs by position and assign order indices
        vias.sort(key=lambda v: v.start if v.start else 0)
        for i, v in enumerate(vias):
            v.order = i

        return NERResult(
            departure=departure,
            arrival=arrival,
            is_valid=is_valid,
            vias=vias,
            departure_start=dep_start,
            departure_end=dep_end,
            arrival_start=arr_start,
            arrival_end=arr_end,
            departure_confidence=dep_confidence,
            arrival_confidence=arr_confidence,
        )

    def _find_cities_in_text(self, text: str, existing_locations: list[dict]) -> list[dict]:
        """
        Find cities in text using dictionary matching with normalization.

        Handles variations like "saint victoret" matching "Saint-Victoret".
        Also handles simple city names like "Paris" matching "Paris Gare de Lyon".
        """
        import re

        locations = list(existing_locations)
        text_lower = text.lower()
        text_normalized = normalize_city_name(text)

        # Common French words that should NOT be matched as cities
        stop_words = {
            "ville", "gare", "train", "centre", "nord", "sud", "est", "ouest",
            "avenue", "rue", "place", "saint", "sainte", "port", "pont",
            "beau", "belle", "grand", "grande", "petit", "petite", "haut", "haute",
            "vieux", "vieille", "nouveau", "nouvelle", "rejoindre", "aller",
            "partir", "arriver", "prendre", "depuis", "vers", "pour", "dans",
            "marne", "seine", "loire", "rhone", "garonne",  # River names often in city names
        }

        # First, try to match full place names (longest first)
        sorted_places = sorted(self.all_places, key=lambda x: len(normalize_city_name(x)), reverse=True)

        for place in sorted_places:
            place_normalized = normalize_city_name(place)

            # Skip very short names (2 chars or less) to avoid false positives like "Eu"
            if len(place_normalized) <= 2:
                continue

            # Skip if the normalized name is a common stop word
            if place_normalized in stop_words:
                continue

            # Try to find normalized version in normalized text
            # Use regex for word boundaries
            pattern = r'\b' + re.escape(place_normalized) + r'\b'
            match = re.search(pattern, text_normalized)

            if match:
                # Find the corresponding position in original text
                norm_start = match.start()
                norm_end = match.end()

                # Map back to original text position
                orig_start = self._map_normalized_pos_to_original(text, text_normalized, norm_start)
                orig_end = self._map_normalized_pos_to_original(text, text_normalized, norm_end)

                if orig_start is not None and orig_end is not None:
                    # Check not overlapping with existing locations
                    overlapping = any(
                        not (orig_end <= loc["start"] or orig_start >= loc["end"])
                        for loc in locations
                    )
                    already_found = any(
                        normalize_city_name(loc["text"]) == place_normalized
                        for loc in locations
                    )

                    if not overlapping and not already_found:
                        locations.append({
                            "text": place,
                            "start": orig_start,
                            "end": orig_end,
                            "confidence": 0.7,
                        })

        # Second pass: find compound city names (e.g., "Nogent sur Marne" -> "Nogent-sur-Marne")
        # Try to match multi-word sequences that could be city names
        if len(locations) < 2:
            # Try to find multi-word city names first (up to 4 words)
            # Match words (letters with optional internal hyphens, but not standalone hyphens)
            word_pattern = r'\b([A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*)\b'
            words = re.findall(word_pattern, text)
            word_positions = [(m.start(), m.end(), m.group(1)) for m in re.finditer(word_pattern, text)]

            for window_size in [4, 3, 2, 1]:
                for i in range(len(word_positions) - window_size + 1):
                    # Build the multi-word candidate
                    start_pos = word_positions[i][0]
                    end_pos = word_positions[i + window_size - 1][1]
                    candidate_words = [word_positions[j][2] for j in range(i, i + window_size)]
                    candidate = " ".join(candidate_words)

                    # Skip if any word is a stop word (for single words only)
                    if window_size == 1 and candidate.lower() in stop_words:
                        continue

                    # Skip very short single words
                    if window_size == 1 and len(candidate) < 4:
                        continue

                    # Check if this region is already covered
                    overlapping = any(
                        not (end_pos <= loc["start"] or start_pos >= loc["end"])
                        for loc in locations
                    )
                    if overlapping:
                        continue

                    # Try to validate as a city
                    validated = self._validate_city(candidate)
                    if validated:
                        # Check we don't already have this city
                        already_found = any(
                            normalize_city_name(loc["text"]).startswith(normalize_city_name(candidate))
                            or normalize_city_name(candidate).startswith(normalize_city_name(loc["text"]))
                            for loc in locations
                        )
                        if not already_found:
                            locations.append({
                                "text": validated,
                                "start": start_pos,
                                "end": end_pos,
                                "confidence": 0.6 + (window_size * 0.05),  # Higher confidence for longer matches
                            })
                            # Found a match, don't look for shorter matches in this region
                            break

        return locations

    def _map_normalized_pos_to_original(self, original: str, normalized: str, norm_pos: int) -> int | None:
        """Map a position in normalized text back to original text."""
        # Simple approach: iterate through both strings simultaneously
        orig_idx = 0
        norm_idx = 0
        orig_lower = original.lower()

        while norm_idx < len(normalized) and orig_idx < len(original):
            if norm_idx == norm_pos:
                return orig_idx

            # Skip chars that were removed in normalization (like hyphens becoming spaces)
            orig_char = orig_lower[orig_idx]
            norm_char = normalized[norm_idx]

            if orig_char == norm_char:
                orig_idx += 1
                norm_idx += 1
            elif orig_char == '-' and norm_char == ' ':
                # Hyphen was replaced by space
                orig_idx += 1
                norm_idx += 1
            elif orig_char == '-':
                # Hyphen was removed
                orig_idx += 1
            elif orig_char in "àâäáã" and norm_char == 'a':
                orig_idx += 1
                norm_idx += 1
            elif orig_char in "éèêë" and norm_char == 'e':
                orig_idx += 1
                norm_idx += 1
            elif orig_char in "îïí" and norm_char == 'i':
                orig_idx += 1
                norm_idx += 1
            elif orig_char in "ôöó" and norm_char == 'o':
                orig_idx += 1
                norm_idx += 1
            elif orig_char in "ùûü" and norm_char == 'u':
                orig_idx += 1
                norm_idx += 1
            elif orig_char == 'ç' and norm_char == 'c':
                orig_idx += 1
                norm_idx += 1
            else:
                # Characters should match after normalization
                orig_idx += 1
                norm_idx += 1

        if norm_idx == norm_pos:
            return orig_idx

        return norm_pos  # Fallback

    def _extract_with_pretrained(self, doc, text: str) -> NERResult:
        """Extract using pretrained LOC labels + rule-based classification."""
        # Get all LOC entities
        locations = []
        for ent in doc.ents:
            if ent.label_ in ("LOC", "GPE"):
                validated = self._validate_city(ent.text)
                if validated:
                    locations.append({
                        "text": validated,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.9,  # spaCy NER detected
                    })

        if len(locations) < 2:
            # Try to find cities by dictionary matching as fallback
            # Also try to match normalized names (e.g., "saint victoret" matches "Saint-Victoret")
            locations = self._find_cities_in_text(text, locations)

        if len(locations) < 2:
            return NERResult(departure=None, arrival=None, is_valid=False)

        # Sort by position
        locations.sort(key=lambda x: x["start"])

        # Classify using context
        departure, arrival, dep_loc, arr_loc = self._classify_locations_with_info(text, locations)

        if departure and arrival and departure != arrival:
            return NERResult(
                departure=departure,
                arrival=arrival,
                is_valid=True,
                departure_start=dep_loc["start"] if dep_loc else None,
                departure_end=dep_loc["end"] if dep_loc else None,
                arrival_start=arr_loc["start"] if arr_loc else None,
                arrival_end=arr_loc["end"] if arr_loc else None,
                departure_confidence=dep_loc.get("confidence", 0.8) if dep_loc else 0.0,
                arrival_confidence=arr_loc.get("confidence", 0.8) if arr_loc else 0.0,
            )

        # Fallback: first is departure, second is arrival
        return NERResult(
            departure=locations[0]["text"],
            arrival=locations[1]["text"],
            is_valid=True,
            departure_start=locations[0]["start"],
            departure_end=locations[0]["end"],
            arrival_start=locations[1]["start"],
            arrival_end=locations[1]["end"],
            departure_confidence=locations[0].get("confidence", 0.6),
            arrival_confidence=locations[1].get("confidence", 0.6),
        )

    def _classify_locations(
        self, text: str, locations: list[dict]
    ) -> tuple[str | None, str | None]:
        """
        Classify locations as departure or arrival based on context.

        Returns:
            (departure, arrival) tuple
        """
        dep, arr, _, _ = self._classify_locations_with_info(text, locations)
        return dep, arr

    def _classify_locations_with_info(
        self, text: str, locations: list[dict]
    ) -> tuple[str | None, str | None, dict | None, dict | None]:
        """
        Classify locations as departure or arrival based on context.

        Uses a priority system to handle ambiguous cases like:
        "Je veux aller de Marseille en partant de Paris"
        where "en partant de" is stronger than simple "de".

        Returns:
            (departure, arrival, dep_location_dict, arr_location_dict) tuple
        """
        text_lower = text.lower()

        # Patterns with priorities (higher = stronger indicator)
        # Format: (pattern, priority)
        dep_patterns = [
            (r"(?:en partant de|partant de|en partance de)\s*$", 10),  # Very strong
            (r"(?:au départ de|au depart de|départ de|depart de)\s*$", 9),
            (r"(?:depuis)\s*$", 8),
            (r"(?:de)\s*$", 2),  # Weak - "aller de X" is often arrival context
        ]

        arr_patterns = [
            (r"(?:aller à|aller a|me rendre à|me rendre a)\s*$", 10),  # Very strong
            (r"(?:vers|direction|pour aller à|pour aller a)\s*$", 9),
            (r"(?:jusqu'à|jusqu'a|rejoindre)\s*$", 8),
            (r"(?:à|a|pour)\s*$", 5),
        ]

        # Score each location for departure and arrival
        dep_scores = {}  # location_text -> (score, loc_dict)
        arr_scores = {}

        for loc in locations:
            before = text_lower[: loc["start"]].strip()
            loc_text = loc["text"]

            # Check departure patterns
            for pattern, priority in dep_patterns:
                if re.search(pattern, before):
                    if loc_text not in dep_scores or priority > dep_scores[loc_text][0]:
                        dep_scores[loc_text] = (priority, loc)
                    break

            # Check arrival patterns
            for pattern, priority in arr_patterns:
                if re.search(pattern, before):
                    if loc_text not in arr_scores or priority > arr_scores[loc_text][0]:
                        arr_scores[loc_text] = (priority, loc)
                    break

        # Special case: "aller de X en partant de Y" means X=arrival, Y=departure
        # If we have "aller de X" (weak dep) and "en partant de Y" (strong dep),
        # X should be reclassified as arrival
        if len(dep_scores) >= 2:
            # Find the strongest departure
            strongest_dep = max(dep_scores.items(), key=lambda x: x[1][0])
            # Reclassify weaker "departures" as arrivals if they used weak pattern
            for loc_text, (score, loc) in list(dep_scores.items()):
                if loc_text != strongest_dep[0] and score <= 2:
                    # Weak departure pattern, likely actually an arrival
                    if loc_text not in arr_scores:
                        arr_scores[loc_text] = (score, loc)
                    del dep_scores[loc_text]

        # Select best departure and arrival
        departure = None
        arrival = None
        dep_loc = None
        arr_loc = None

        if dep_scores:
            best_dep = max(dep_scores.items(), key=lambda x: x[1][0])
            departure = best_dep[0]
            dep_loc = best_dep[1][1]

        if arr_scores:
            # Exclude the departure from arrival candidates
            valid_arr = {k: v for k, v in arr_scores.items() if k != departure}
            if valid_arr:
                best_arr = max(valid_arr.items(), key=lambda x: x[1][0])
                arrival = best_arr[0]
                arr_loc = best_arr[1][1]

        return departure, arrival, dep_loc, arr_loc

    def extract(self, sentence: str) -> NERResult:
        """
        Extract departure and arrival from a sentence.

        Args:
            sentence: Input sentence

        Returns:
            NERResult with departure, arrival, and validity
        """
        doc = self.nlp(sentence)

        if self.custom_model:
            return self._extract_with_custom_model(doc)
        else:
            return self._extract_with_pretrained(doc, sentence)

    def process_batch(self, sentences: list[str]) -> list[NERResult]:
        """Process multiple sentences efficiently using nlp.pipe()."""
        results = []
        for doc, sentence in zip(self.nlp.pipe(sentences), sentences):
            if self.custom_model:
                results.append(self._extract_with_custom_model(doc))
            else:
                results.append(self._extract_with_pretrained(doc, sentence))
        return results
