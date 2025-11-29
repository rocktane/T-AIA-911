"""Text preprocessing utilities."""

import re
import unicodedata

from rapidfuzz import fuzz, process


# Common French abbreviations for city names
ABBREVIATIONS = {
    "st": "saint",
    "ste": "sainte",
    "mt": "mont",
    "mte": "monte",
    "pt": "pont",
    "gd": "grand",
    "gde": "grande",
    "pts": "ponts",
}

# Compile regex patterns for word boundaries
ABBREV_PATTERNS = {
    abbrev: re.compile(rf"\b{abbrev}\b", re.IGNORECASE)
    for abbrev in ABBREVIATIONS
}


def remove_accents(text: str) -> str:
    """Remove accents from text while preserving case."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace (collapse multiple spaces, strip)."""
    return " ".join(text.split())


def expand_abbreviations(text: str) -> str:
    """
    Expand common French abbreviations in city names.

    Examples:
        "st victoret" -> "saint victoret"
        "ste marie" -> "sainte marie"
        "mt blanc" -> "mont blanc"

    Handles word boundaries to avoid transforming "stable" -> "saintable".
    """
    result = text
    for abbrev, expansion in ABBREVIATIONS.items():
        result = ABBREV_PATTERNS[abbrev].sub(expansion, result)
    return result


def preprocess_text(text: str, lowercase: bool = False, remove_accents_flag: bool = False) -> str:
    """
    Preprocess text for NLP processing.

    Args:
        text: Input text
        lowercase: Convert to lowercase
        remove_accents_flag: Remove accents

    Returns:
        Preprocessed text
    """
    # Normalize whitespace
    text = normalize_whitespace(text)

    # Optionally lowercase
    if lowercase:
        text = text.lower()

    # Optionally remove accents
    if remove_accents_flag:
        text = remove_accents(text)

    return text


def normalize_city_name(name: str) -> str:
    """
    Normalize a city name for matching.

    Converts to lowercase, removes accents, expands abbreviations,
    replaces hyphens with spaces, and normalizes whitespace.

    Examples:
        "Saint-Victoret" -> "saint victoret"
        "ST VICTORET" -> "saint victoret"
        "st-victoret" -> "saint victoret"
    """
    name = name.lower()
    name = remove_accents(name)
    # Replace hyphens with spaces for matching
    name = name.replace("-", " ")
    # Expand abbreviations (st -> saint, etc.)
    name = expand_abbreviations(name)
    name = normalize_whitespace(name)
    return name


def extract_potential_cities(text: str, city_set: set[str]) -> list[str]:
    """
    Extract potential city names from text using simple matching.

    Args:
        text: Input text
        city_set: Set of known city names (normalized)

    Returns:
        List of matched city names
    """
    text_normalized = normalize_city_name(text)
    found = []

    for city in city_set:
        city_normalized = normalize_city_name(city)
        if city_normalized in text_normalized:
            found.append(city)

    # Sort by length (longer matches first) to handle overlapping names
    found.sort(key=len, reverse=True)
    return found


def fuzzy_match_city(
    text: str,
    city_list: list[str],
    threshold: int = 85,
    limit: int = 1,
) -> list[tuple[str, int]]:
    """
    Find the best matching city name using fuzzy matching.

    Useful for correcting typos like "Marseile" -> "Marseille".

    Args:
        text: Input text (potential city name with typo)
        city_list: List of known city names
        threshold: Minimum similarity score (0-100)
        limit: Maximum number of matches to return

    Returns:
        List of tuples (city_name, score) sorted by score descending
    """
    if not text or not city_list:
        return []

    text_normalized = normalize_city_name(text)

    # Build normalized lookup
    normalized_to_original = {}
    normalized_cities = []
    for city in city_list:
        norm = normalize_city_name(city)
        normalized_to_original[norm] = city
        normalized_cities.append(norm)

    # Use rapidfuzz for efficient fuzzy matching
    results = process.extract(
        text_normalized,
        normalized_cities,
        scorer=fuzz.ratio,
        limit=limit,
        score_cutoff=threshold,
    )

    # Map back to original city names
    matches = []
    for match_norm, score, _idx in results:
        original = normalized_to_original.get(match_norm, match_norm)
        matches.append((original, int(score)))

    return matches


def correct_city_typo(
    text: str,
    city_list: list[str],
    threshold: int = 85,
) -> str | None:
    """
    Correct a potential typo in a city name.

    Args:
        text: Input text (potential city name with typo)
        city_list: List of known city names
        threshold: Minimum similarity score (0-100)

    Returns:
        Corrected city name or None if no good match found

    Examples:
        correct_city_typo("Marseile", cities) -> "Marseille"
        correct_city_typo("Bordeau", cities) -> "Bordeaux"
        correct_city_typo("Toulous", cities) -> "Toulouse"
    """
    matches = fuzzy_match_city(text, city_list, threshold=threshold, limit=1)
    if matches:
        return matches[0][0]
    return None


def find_cities_with_fuzzy(
    text: str,
    city_list: list[str],
    threshold: int = 85,
) -> list[tuple[str, str, int]]:
    """
    Find potential city names in text using fuzzy matching.

    Extracts words/phrases and tries to match them against known cities.

    Args:
        text: Input text
        city_list: List of known city names
        threshold: Minimum similarity score (0-100)

    Returns:
        List of tuples (original_text, matched_city, score)
    """
    # Extract potential city tokens (words and multi-word combinations)
    words = text.split()
    candidates = []

    # Single words
    candidates.extend(words)

    # Two-word combinations (for "Saint Pierre", etc.)
    for i in range(len(words) - 1):
        candidates.append(f"{words[i]} {words[i+1]}")

    # Three-word combinations (for "La Roche sur", etc.)
    for i in range(len(words) - 2):
        candidates.append(f"{words[i]} {words[i+1]} {words[i+2]}")

    results = []
    seen_cities = set()

    for candidate in candidates:
        if len(candidate) < 3:  # Skip very short tokens
            continue

        matches = fuzzy_match_city(candidate, city_list, threshold=threshold, limit=1)
        for city, score in matches:
            if city not in seen_cities:
                results.append((candidate, city, score))
                seen_cities.add(city)

    # Sort by score descending
    results.sort(key=lambda x: x[2], reverse=True)
    return results
