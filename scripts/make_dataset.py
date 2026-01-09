"""
Dataset Generator for Travel Order Resolver

Generates training and test datasets for NER model training.
Target: ~10,000 sentences with ~100 different grammatical structures.

Output formats:
- CSV: sentenceID,sentence,departure,destination,is_valid
- spaCy DocBin: train.spacy, test.spacy

Templates are loaded from external files in data/templates/ for easy editing.
"""

import csv
import random
import re
import unicodedata
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# Configuration
SEED = 42
N_VALID_NO_VIA = 5000        # 50% - Classic 2-city queries (backward compat)
N_VALID_SINGLE_VIA = 2500    # 25% - Single VIA point
N_VALID_MULTI_VIA = 1500     # 15% - 2-3 VIA points
N_INVALID_SENTENCES = 1000   # 10% - Invalid sentences
TRAIN_RATIO = 0.8
DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
OUTPUT_DIR = DATA_DIR

# Place source configuration (stations vs communes vs mixed)
# - stations: use SNCF station names (gares-france.csv)
# - cities: use commune names (communes-france.csv)
# - mixed: sample stations/cities with CITY_PROB
PLACE_SOURCE_MODE = "mixed"  # "stations" | "cities" | "mixed"
CITY_PROB = 0.5
# Loading all communes can be large; cap to keep generation fast.
MAX_CITY_NAMES = 50000

# Entity marker tokens (must survive lowercasing / accent stripping)
_DEP_OPEN, _DEP_CLOSE = "⟦0⟧", "⟦/0⟧"
_ARR_OPEN, _ARR_CLOSE = "⟦1⟧", "⟦/1⟧"
_V1_OPEN, _V1_CLOSE = "⟦2⟧", "⟦/2⟧"
_V2_OPEN, _V2_CLOSE = "⟦3⟧", "⟦/3⟧"
_V3_OPEN, _V3_CLOSE = "⟦4⟧", "⟦/4⟧"


def _wrap(value: str, open_tok: str, close_tok: str) -> str:
    return f"{open_tok}{value}{close_tok}"


def strip_entity_markers(marked_text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """
    Strip marker tokens from a sentence and return (clean_text, spans).

    Spans are tuples: (start, end, label) in clean_text coordinates.
    """
    tokens = [
        (_DEP_OPEN, _DEP_CLOSE, "DEPART"),
        (_ARR_OPEN, _ARR_CLOSE, "ARRIVEE"),
        (_V1_OPEN, _V1_CLOSE, "VIA"),
        (_V2_OPEN, _V2_CLOSE, "VIA"),
        (_V3_OPEN, _V3_CLOSE, "VIA"),
    ]

    out_chars: list[str] = []
    spans: list[tuple[int, int, str]] = []
    active: tuple[str, str, str] | None = None
    active_start: int | None = None

    i = 0
    clean_idx = 0
    while i < len(marked_text):
        # Close marker (if we are inside an entity)
        if active is not None and marked_text.startswith(active[1], i):
            if active_start is None:
                raise ValueError("marker close without start")
            spans.append((active_start, clean_idx, active[2]))
            i += len(active[1])
            active = None
            active_start = None
            continue

        # Open marker
        opened = False
        for open_tok, close_tok, label in tokens:
            if marked_text.startswith(open_tok, i):
                if active is not None:
                    raise ValueError("nested entity markers are not supported")
                active = (open_tok, close_tok, label)
                active_start = clean_idx
                i += len(open_tok)
                opened = True
                break
        if opened:
            continue

        out_chars.append(marked_text[i])
        i += 1
        clean_idx += 1

    if active is not None:
        raise ValueError("unterminated entity marker")

    clean_text = "".join(out_chars)
    return clean_text, spans


def load_templates(filepath: Path) -> list[str]:
    """Load templates from a text file.

    Lines starting with # are comments.
    Empty lines are ignored.
    """
    templates = []
    if not filepath.exists():
        print(f"Warning: Template file not found: {filepath}")
        return templates

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                templates.append(line)

    return templates


def load_all_templates() -> dict:
    """Load all templates from the templates directory."""
    return {
        "dep_first": load_templates(TEMPLATES_DIR / "departure_first.txt"),
        "arr_first": load_templates(TEMPLATES_DIR / "arrival_first.txt"),
        "via_single": load_templates(TEMPLATES_DIR / "via.txt"),
        "via_multi": load_templates(TEMPLATES_DIR / "via_multi.txt"),
        "invalid": load_templates(TEMPLATES_DIR / "invalid.txt"),
        "ambiguous": load_templates(TEMPLATES_DIR / "ambiguous.txt"),
        "name_city_contexts": load_templates(TEMPLATES_DIR / "name_city_contexts.txt"),
        "name_cities": load_templates(TEMPLATES_DIR / "name_cities.txt"),
        "compound_cities": load_templates(TEMPLATES_DIR / "compound_cities.txt"),
    }

def _dedupe_preserve_order(items: list[str]) -> tuple[list[str], int]:
    """Remove exact duplicates, preserving original order."""
    out: list[str] = []
    seen: set[str] = set()
    removed = 0
    for s in items:
        if s in seen:
            removed += 1
            continue
        seen.add(s)
        out.append(s)
    return out, removed


_PLACEHOLDERS = ("{D}", "{A}", "{V}", "{V1}", "{V2}", "{V3}", "{N}")


def _count_placeholders(template: str) -> dict[str, int]:
    return {p: template.count(p) for p in _PLACEHOLDERS}


def _has_adjacent_placeholders(template: str) -> bool:
    # e.g. "...{D}{A}..." or "...{A}{D}..."
    return bool(re.search(r"\}\{", template))


def validate_and_dedupe_templates(templates: dict) -> dict:
    """
    Validate templates and remove exact duplicates.

    This keeps template files scalable as we add hundreds of lines:
    - removes exact duplicates (per category)
    - warns/drops templates with missing or repeated placeholders
    - prevents placeholders from appearing in invalid templates
    """
    out = dict(templates)

    # Dedupe per category
    for key, items in list(out.items()):
        if not isinstance(items, list):
            continue
        deduped, removed = _dedupe_preserve_order(items)
        if removed:
            print(f"  [templates] Removed {removed} duplicate lines from {key}")
        out[key] = deduped

    def warn(key: str, template: str, reason: str) -> None:
        print(f"  [templates] Warning in {key}: {reason}: {template}")

    filtered: dict[str, list[str]] = {}
    for key, items in out.items():
        if not isinstance(items, list):
            filtered[key] = items
            continue

        ok: list[str] = []
        for t in items:
            counts = _count_placeholders(t)
            if _has_adjacent_placeholders(t):
                warn(key, t, "adjacent placeholders")

            if key in ("dep_first", "arr_first"):
                if counts["{D}"] != 1 or counts["{A}"] != 1:
                    warn(key, t, "expected exactly one {D} and one {A}")
                    continue
                if any(counts[p] for p in ("{V}", "{V1}", "{V2}", "{V3}")):
                    warn(key, t, "unexpected VIA placeholder in non-VIA template")
                    continue

            elif key == "via_single":
                if counts["{D}"] != 1 or counts["{A}"] != 1 or counts["{V}"] != 1:
                    warn(key, t, "expected exactly one {D}, one {A}, one {V}")
                    continue
                if any(counts[p] for p in ("{V1}", "{V2}", "{V3}")):
                    warn(key, t, "unexpected {V1..V3} in single-VIA template")
                    continue

            elif key == "via_multi":
                if counts["{D}"] != 1 or counts["{A}"] != 1:
                    warn(key, t, "expected exactly one {D} and one {A}")
                    continue
                if counts["{V}"]:
                    warn(key, t, "unexpected {V} in multi-VIA template")
                    continue
                if counts["{V1}"] != 1:
                    warn(key, t, "expected exactly one {V1}")
                    continue
                if counts["{V2}"] not in (0, 1) or counts["{V3}"] not in (0, 1):
                    warn(key, t, "expected {V2}/{V3} to appear 0 or 1 times")
                    continue

            elif key == "ambiguous":
                if (counts["{D}"] + counts["{A}"]) != 1:
                    warn(key, t, "expected exactly one of {D} or {A}")
                    continue
                if any(counts[p] for p in ("{V}", "{V1}", "{V2}", "{V3}", "{N}")):
                    warn(key, t, "unexpected placeholder in ambiguous template")
                    continue

            elif key == "invalid":
                if any(counts[p] for p in _PLACEHOLDERS):
                    warn(key, t, "invalid templates must not contain placeholders")
                    continue

            ok.append(t)

        filtered[key] = ok

    return filtered


def load_stations(filepath: Path) -> list[str]:
    """Load station names from CSV file."""
    stations = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["nom"].strip()
            if name and len(name) > 1:
                stations.append(name)
    return list(set(stations))


def load_communes(filepath: Path, max_names: int | None = None) -> list[str]:
    """Load commune (city) names from CSV file."""
    cities: list[str] = []
    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # commons: nom_commune, latitude, longitude, ...
            name = (row.get("nom_commune") or "").strip()
            if name and len(name) > 1:
                cities.append(name)
            if max_names is not None and len(cities) >= max_names:
                break
    return list(set(cities))


def choose_place(stations: list[str], cities: list[str]) -> str:
    """Choose a place name from stations/cities depending on config."""
    if PLACE_SOURCE_MODE == "stations" or not cities:
        return random.choice(stations)
    if PLACE_SOURCE_MODE == "cities":
        return random.choice(cities)
    # mixed
    if random.random() < CITY_PROB:
        return random.choice(cities)
    return random.choice(stations)


def choose_distinct_places(
    count: int, stations: list[str], cities: list[str], *, excluded: set[str] | None = None
) -> list[str] | None:
    """Pick N distinct places. Returns None if it fails too often."""
    excluded = excluded or set()
    picked: list[str] = []
    used = set(excluded)
    attempts = 0
    max_attempts = 200
    while len(picked) < count and attempts < max_attempts:
        attempts += 1
        p = choose_place(stations, cities)
        if p in used:
            continue
        used.add(p)
        picked.append(p)
    if len(picked) != count:
        return None
    return picked


def violates_ordered_substring_collision(template: str, values: dict[str, str]) -> bool:
    """
    Prevent cases where a later placeholder value is a substring of an earlier one.

    This was a major source of span-alignment failures with naive `text.find()`
    (e.g. D=\"Lyon\" inside A=\"Paris Gare de Lyon\" when {A} comes before {D}).
    Even with marker-based alignment, it reduces ambiguous training examples.
    """
    positions: list[tuple[int, str, str]] = []
    for placeholder, value in values.items():
        pos = template.find(placeholder)
        if pos != -1:
            positions.append((pos, placeholder, value))

    for pos_i, _ph_i, val_i in positions:
        val_i_low = val_i.lower().strip()
        if len(val_i_low) < 3:
            continue
        for pos_j, _ph_j, val_j in positions:
            if pos_i <= pos_j:
                continue
            val_j_low = val_j.lower().strip()
            if val_i_low and val_i_low in val_j_low:
                return True
    return False


def remove_accents(text: str) -> str:
    """Remove accents from text."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def add_typo(word: str) -> str:
    """Add a random typo to a word."""
    if len(word) < 4:
        return word

    typo_type = random.choice(["double", "swap", "remove", "replace"])

    if typo_type == "double" and len(word) > 3:
        # Double a letter
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx] + word[idx:]

    elif typo_type == "swap" and len(word) > 3:
        # Swap two adjacent letters
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1] + word[idx] + word[idx + 2 :]

    elif typo_type == "remove" and len(word) > 4:
        # Remove a letter
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx + 1 :]

    elif typo_type == "replace":
        # Replace a letter with a nearby key
        nearby = {
            "a": "qzs",
            "e": "rz",
            "i": "ou",
            "o": "ip",
            "u": "yi",
            "p": "ol",
            "s": "dqa",
            "l": "mk",
            "m": "nl",
            "n": "bm",
        }
        idx = random.randint(1, len(word) - 2)
        char = word[idx].lower()
        if char in nearby:
            new_char = random.choice(nearby[char])
            return word[:idx] + new_char + word[idx + 1 :]

    return word


def apply_noise(text: str, dep: str, arr: str) -> tuple[str, str, str]:
    """Apply various noise transformations to text and city names."""
    noise_type = random.random()

    if noise_type < 0.15:
        # Lowercase everything
        return text.lower(), dep.lower(), arr.lower()

    elif noise_type < 0.25:
        # Remove accents
        return remove_accents(text), remove_accents(dep), remove_accents(arr)

    elif noise_type < 0.30:
        # Lowercase + no accents
        text = remove_accents(text.lower())
        return text, remove_accents(dep.lower()), remove_accents(arr.lower())

    elif noise_type < 0.35:
        # Typo in departure
        dep_typo = add_typo(dep)
        return text.replace(dep, dep_typo), dep_typo, arr

    elif noise_type < 0.40:
        # Typo in arrival
        arr_typo = add_typo(arr)
        return text.replace(arr, arr_typo), dep, arr_typo

    # No noise (60% of cases)
    return text, dep, arr


def apply_noise_with_vias(
    text: str, dep: str, arr: str, vias: list[str]
) -> tuple[str, str, str, list[str]]:
    """Apply various noise transformations to text and city names including VIAs."""
    noise_type = random.random()

    if noise_type < 0.15:
        # Lowercase everything
        return text.lower(), dep.lower(), arr.lower(), [v.lower() for v in vias]

    elif noise_type < 0.25:
        # Remove accents
        return (
            remove_accents(text),
            remove_accents(dep),
            remove_accents(arr),
            [remove_accents(v) for v in vias],
        )

    elif noise_type < 0.30:
        # Lowercase + no accents
        text = remove_accents(text.lower())
        return (
            text,
            remove_accents(dep.lower()),
            remove_accents(arr.lower()),
            [remove_accents(v.lower()) for v in vias],
        )

    elif noise_type < 0.35:
        # Typo in departure
        dep_typo = add_typo(dep)
        return text.replace(dep, dep_typo), dep_typo, arr, vias

    elif noise_type < 0.40:
        # Typo in arrival
        arr_typo = add_typo(arr)
        return text.replace(arr, arr_typo), dep, arr_typo, vias

    elif noise_type < 0.45 and vias:
        # Typo in a random VIA
        via_idx = random.randint(0, len(vias) - 1)
        via_typo = add_typo(vias[via_idx])
        new_vias = vias.copy()
        text = text.replace(vias[via_idx], via_typo)
        new_vias[via_idx] = via_typo
        return text, dep, arr, new_vias

    # No noise (55-60% of cases)
    return text, dep, arr, vias


def generate_valid_sentences(
    stations: list[str], cities: list[str], n_samples: int, templates: dict
) -> list[dict]:
    """Generate valid travel order sentences (DEPART + ARRIVEE)."""
    sentences: list[dict] = []
    seen: set[str] = set()

    all_templates = templates["dep_first"] + templates["arr_first"]
    name_cities = templates["name_cities"]
    name_city_contexts = templates["name_city_contexts"]
    compound_cities = templates.get("compound_cities", [])

    places = stations + cities
    name_cities_in_places = [c for c in name_cities if c in places]

    # Compound places that exist in pool (exact match or prefix)
    compound_places_in_pool: list[str] = []
    for compound in compound_cities:
        compound_lower = compound.lower()
        for place in places:
            place_lower = place.lower()
            if place_lower == compound_lower or place_lower.startswith(compound_lower + " "):
                compound_places_in_pool.append(place)
                break

    pbar = tqdm(total=n_samples, desc="Generating valid sentences")

    while len(sentences) < n_samples:
        rand_val = random.random()

        # 10% chance: name-city contexts (N is NOT a location entity)
        if rand_val < 0.10 and name_cities_in_places and name_city_contexts:
            template = random.choice(name_city_contexts)
            name = random.choice(name_cities)
            picked = choose_distinct_places(2, stations, cities, excluded={name})
            if not picked:
                continue
            dep, arr = picked
            if violates_ordered_substring_collision(template, {"{D}": dep, "{A}": arr}):
                continue
            marked = template.format(
                N=name,
                D=_wrap(dep, _DEP_OPEN, _DEP_CLOSE),
                A=_wrap(arr, _ARR_OPEN, _ARR_CLOSE),
            )

        # 15% chance: force a compound place for dep or arr
        elif rand_val < 0.25 and compound_places_in_pool and all_templates:
            template = random.choice(all_templates)
            if random.random() < 0.5:
                dep = random.choice(compound_places_in_pool)
                arr = choose_place(stations, cities)
                if arr == dep:
                    continue
            else:
                dep = choose_place(stations, cities)
                arr = random.choice(compound_places_in_pool)
                if arr == dep:
                    continue
            if violates_ordered_substring_collision(template, {"{D}": dep, "{A}": arr}):
                continue
            marked = template.format(
                D=_wrap(dep, _DEP_OPEN, _DEP_CLOSE),
                A=_wrap(arr, _ARR_OPEN, _ARR_CLOSE),
            )

        else:
            template = random.choice(all_templates)
            picked = choose_distinct_places(2, stations, cities)
            if not picked:
                continue
            dep, arr = picked
            if violates_ordered_substring_collision(template, {"{D}": dep, "{A}": arr}):
                continue
            marked = template.format(
                D=_wrap(dep, _DEP_OPEN, _DEP_CLOSE),
                A=_wrap(arr, _ARR_OPEN, _ARR_CLOSE),
            )

        # Apply noise (lowercase, no accents, typos) on marked sentence
        marked_noisy, _dep_noisy, _arr_noisy = apply_noise(marked, dep, arr)

        try:
            text, spans = strip_entity_markers(marked_noisy)
        except ValueError:
            continue

        if text in seen:
            continue

        # Must contain exactly one DEPART and one ARRIVEE
        if sum(1 for _s, _e, lab in spans if lab == "DEPART") != 1:
            continue
        if sum(1 for _s, _e, lab in spans if lab == "ARRIVEE") != 1:
            continue

        seen.add(text)
        sentences.append(
            {
                "sentence": text,
                "departure": dep,
                "arrival": arr,
                "vias": [],
                "spans": spans,
                "is_valid": True,
            }
        )
        pbar.update(1)

    pbar.close()
    return sentences


def generate_via_sentences(
    stations: list[str],
    cities: list[str],
    n_single_via: int,
    n_multi_via: int,
    templates: dict,
) -> list[dict]:
    """Generate valid travel sentences with VIA waypoints."""
    sentences: list[dict] = []
    seen: set[str] = set()
    via_single_templates = templates.get("via_single", [])
    via_multi_templates = templates.get("via_multi", [])

    if not via_single_templates:
        print("Warning: No single VIA templates found")
        return sentences

    pbar = tqdm(total=n_single_via + n_multi_via, desc="Generating VIA sentences")

    # Generate single VIA sentences
    attempts = 0
    max_attempts = n_single_via * 20
    while len(sentences) < n_single_via and attempts < max_attempts:
        attempts += 1
        template = random.choice(via_single_templates)

        # Select distinct cities
        picked = choose_distinct_places(3, stations, cities)
        if not picked:
            continue
        dep, arr, via = picked

        if violates_ordered_substring_collision(template, {"{D}": dep, "{A}": arr, "{V}": via}):
            continue

        try:
            marked = template.format(
                D=_wrap(dep, _DEP_OPEN, _DEP_CLOSE),
                A=_wrap(arr, _ARR_OPEN, _ARR_CLOSE),
                V=_wrap(via, _V1_OPEN, _V1_CLOSE),
            )
        except KeyError:
            continue

        # Apply noise
        marked_noisy, _dep_noisy, _arr_noisy, _vias_noisy = apply_noise_with_vias(
            marked, dep, arr, [via]
        )

        try:
            text, spans = strip_entity_markers(marked_noisy)
        except ValueError:
            continue

        if text in seen:
            continue
        seen.add(text)
        sentences.append({
            "sentence": text,
            "departure": dep,
            "arrival": arr,
            "vias": [via],
            "spans": spans,
            "is_valid": True,
        })
        pbar.update(1)

    # Generate multi-VIA sentences (2-3 VIA points)
    if via_multi_templates:
        attempts = 0
        max_attempts = n_multi_via * 20
        while len(sentences) < n_single_via + n_multi_via and attempts < max_attempts:
            attempts += 1
            template = random.choice(via_multi_templates)

            # Determine number of VIAs based on template
            has_v3 = "{V3}" in template
            has_v2 = "{V2}" in template
            has_v1 = "{V1}" in template

            # Select distinct cities
            need = 2 + (1 if has_v1 else 0) + (1 if has_v2 else 0) + (1 if has_v3 else 0)
            picked = choose_distinct_places(need, stations, cities)
            if not picked:
                continue
            dep = picked[0]
            arr = picked[1]
            vias = picked[2:]

            collision_values = {"{D}": dep, "{A}": arr}
            if has_v1 and len(vias) >= 1:
                collision_values["{V1}"] = vias[0]
            if has_v2 and len(vias) >= 2:
                collision_values["{V2}"] = vias[1]
            if has_v3 and len(vias) >= 3:
                collision_values["{V3}"] = vias[2]
            if violates_ordered_substring_collision(template, collision_values):
                continue

            try:
                format_dict = {}
                if len(vias) >= 1:
                    format_dict["V1"] = _wrap(vias[0], _V1_OPEN, _V1_CLOSE)
                if len(vias) >= 2:
                    format_dict["V2"] = _wrap(vias[1], _V2_OPEN, _V2_CLOSE)
                if len(vias) >= 3:
                    format_dict["V3"] = _wrap(vias[2], _V3_OPEN, _V3_CLOSE)
                marked = template.format(
                    D=_wrap(dep, _DEP_OPEN, _DEP_CLOSE),
                    A=_wrap(arr, _ARR_OPEN, _ARR_CLOSE),
                    **format_dict,
                )
            except KeyError:
                continue

            # Apply noise
            marked_noisy, _dep_noisy, _arr_noisy, _vias_noisy = apply_noise_with_vias(
                marked, dep, arr, vias
            )

            try:
                text, spans = strip_entity_markers(marked_noisy)
            except ValueError:
                continue

            if text in seen:
                continue
            seen.add(text)
            sentences.append({
                "sentence": text,
                "departure": dep,
                "arrival": arr,
                "vias": vias,
                "spans": spans,
                "is_valid": True,
            })
            pbar.update(1)

    pbar.close()
    return sentences


def generate_invalid_sentences(
    stations: list[str], cities: list[str], n_samples: int, templates: dict
) -> list[dict]:
    """Generate invalid (non-travel) sentences."""
    sentences: list[dict] = []
    seen: set[str] = set()
    invalid_templates = templates["invalid"]
    ambiguous_templates = templates["ambiguous"]

    # Split: 30% completely invalid, 70% ambiguous (reversed to have more variety)
    # Ambiguous sentences have more variations because they use city names
    n_invalid = min(int(n_samples * 0.3), len(invalid_templates) * 4)  # Cap at 4 variants per template
    n_ambiguous = n_samples - n_invalid

    pbar = tqdm(total=n_samples, desc="Generating invalid sentences")

    # Generate all possible invalid variations first
    all_invalid_variants = []
    for text in invalid_templates:
        # Original
        all_invalid_variants.append(text)
        # Lowercase
        all_invalid_variants.append(text.lower())
        # No accents
        all_invalid_variants.append(remove_accents(text))
        # Lowercase + no accents
        all_invalid_variants.append(remove_accents(text.lower()))

    # Remove duplicates and shuffle
    all_invalid_variants = list(set(all_invalid_variants))
    random.shuffle(all_invalid_variants)

    # Completely invalid sentences - take up to n_invalid
    for text in all_invalid_variants[:n_invalid]:
        if text not in seen:
            seen.add(text)
            sentences.append({
                "sentence": text,
                "departure": "",
                "arrival": "",
                "vias": [],
                "spans": [],
                "is_valid": False,
            })
            pbar.update(1)

    # Ambiguous sentences - these have more variety with city names
    attempts = 0
    max_attempts = n_ambiguous * 10  # Prevent infinite loop
    while len(sentences) < n_samples and attempts < max_attempts:
        attempts += 1
        template = random.choice(ambiguous_templates)
        city = choose_place(stations, cities)

        if "{D}" in template:
            text = template.format(D=city)
        elif "{A}" in template:
            text = template.format(A=city)
        else:
            text = template

        # Apply noise variations
        noise = random.random()
        if noise < 0.25:
            text = text.lower()
        elif noise < 0.40:
            text = remove_accents(text)
        elif noise < 0.50:
            text = remove_accents(text.lower())

        if text in seen:
            continue

        seen.add(text)
        sentences.append({
            "sentence": text,
            "departure": "",
            "arrival": "",
            "vias": [],
            "spans": [],
            "is_valid": False,
        })
        pbar.update(1)

    pbar.close()
    return sentences


def save_to_csv(
    sentences: list[dict], filepath: Path
) -> None:
    """Save sentences to CSV file with VIA cities."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentenceID", "sentence", "departure", "destination", "vias", "is_valid"])
        for i, s in enumerate(sentences, 1):
            # Join VIA cities with pipe separator
            vias_str = "|".join(s.get("vias", []))
            writer.writerow([
                i,
                s["sentence"],
                s["departure"],
                s["arrival"],
                vias_str,
                s["is_valid"],
            ])
    print(f"Saved {len(sentences)} sentences to {filepath}")


def save_to_spacy(
    sentences: list[dict], filepath: Path
) -> None:
    """Save sentences to spaCy DocBin format with DEPART, ARRIVEE, and VIA entities."""
    nlp = spacy.blank("fr")
    db = DocBin()
    skipped = 0

    for s in tqdm(sentences, desc=f"Creating {filepath.name}"):
        text = s["sentence"]
        doc = nlp.make_doc(text)

        if not s["is_valid"]:
            # No entities for invalid sentences
            doc.ents = []
            db.add(doc)
            continue

        spans = s.get("spans", [])
        if not spans:
            skipped += 1
            continue

        # Must contain exactly one DEPART and one ARRIVEE (VIA optional)
        if sum(1 for _s, _e, lab in spans if lab == "DEPART") != 1:
            skipped += 1
            continue
        if sum(1 for _s, _e, lab in spans if lab == "ARRIVEE") != 1:
            skipped += 1
            continue

        entities = []
        entity_ranges: list[tuple[int, int]] = []
        valid = True
        for start, end, label in spans:
            span = doc.char_span(start, end, label=label)
            if span is None:
                valid = False
                break
            # Prevent overlap
            for existing_start, existing_end in entity_ranges:
                if not (end <= existing_start or existing_end <= start):
                    valid = False
                    break
            if not valid:
                break
            entities.append(span)
            entity_ranges.append((start, end))

        if not valid:
            skipped += 1
            continue

        entities.sort(key=lambda sp: sp.start_char)
        doc.ents = entities
        db.add(doc)

    db.to_disk(filepath)
    print(f"Saved {len(sentences) - skipped} sentences to {filepath} (skipped {skipped})")


def main():
    random.seed(SEED)

    # Load templates from files
    print("Loading templates...")
    templates = validate_and_dedupe_templates(load_all_templates())

    total_templates = (
        len(templates["dep_first"])
        + len(templates["arr_first"])
        + len(templates.get("via_single", []))
        + len(templates.get("via_multi", []))
    )
    print(f"  Departure-first templates: {len(templates['dep_first'])}")
    print(f"  Arrival-first templates: {len(templates['arr_first'])}")
    print(f"  VIA single templates: {len(templates.get('via_single', []))}")
    print(f"  VIA multi templates: {len(templates.get('via_multi', []))}")
    print(f"  Invalid sentences: {len(templates['invalid'])}")
    print(f"  Ambiguous templates: {len(templates['ambiguous'])}")
    print(f"  Name-city contexts: {len(templates['name_city_contexts'])}")
    print(f"  Name-cities: {len(templates['name_cities'])}")
    print(f"  Compound cities: {len(templates.get('compound_cities', []))}")

    # Load stations from CSV
    stations_file = DATA_DIR / "gares-france.csv"
    if not stations_file.exists():
        print(f"Error: {stations_file} not found")
        print("Please download station data from https://ressources.data.sncf.com/")
        return

    stations = load_stations(stations_file)
    print(f"\nLoaded {len(stations)} stations from {stations_file}")

    # Load communes (cities) from CSV (optional but recommended for mixed mode)
    cities_file = DATA_DIR / "communes-france.csv"
    cities: list[str] = []
    if cities_file.exists():
        cities = load_communes(cities_file, max_names=MAX_CITY_NAMES)
        print(f"Loaded {len(cities)} cities from {cities_file} (cap={MAX_CITY_NAMES})")
    else:
        print(f"Warning: {cities_file} not found (cities mode disabled)")

    # Generate sentences
    print("\n--- Generating sentences ---")
    valid_no_via = generate_valid_sentences(stations, cities, N_VALID_NO_VIA, templates)
    via_sentences = generate_via_sentences(
        stations, cities, N_VALID_SINGLE_VIA, N_VALID_MULTI_VIA, templates
    )
    invalid_sentences = generate_invalid_sentences(stations, cities, N_INVALID_SENTENCES, templates)

    # Combine all valid sentences
    all_valid = valid_no_via + via_sentences

    # Combine and shuffle
    all_sentences = all_valid + invalid_sentences
    random.shuffle(all_sentences)

    # Count VIA sentences
    n_single_via = sum(1 for s in via_sentences if len(s.get("vias", [])) == 1)
    n_multi_via = sum(1 for s in via_sentences if len(s.get("vias", [])) > 1)

    print(f"\nTotal sentences: {len(all_sentences)}")
    print(f"  Valid (no VIA): {len(valid_no_via)} ({len(valid_no_via)/len(all_sentences)*100:.1f}%)")
    print(f"  Valid (single VIA): {n_single_via} ({n_single_via/len(all_sentences)*100:.1f}%)")
    print(f"  Valid (multi VIA): {n_multi_via} ({n_multi_via/len(all_sentences)*100:.1f}%)")
    print(f"  Invalid: {len(invalid_sentences)} ({len(invalid_sentences)/len(all_sentences)*100:.1f}%)")

    # Split into train/test
    split_idx = int(len(all_sentences) * TRAIN_RATIO)
    train_sentences = all_sentences[:split_idx]
    test_sentences = all_sentences[split_idx:]

    print(f"\nTrain set: {len(train_sentences)}")
    print(f"Test set: {len(test_sentences)}")

    # Save to CSV
    save_to_csv(train_sentences, OUTPUT_DIR / "dataset_train.csv")
    save_to_csv(test_sentences, OUTPUT_DIR / "dataset_test.csv")
    save_to_csv(all_sentences, OUTPUT_DIR / "dataset_full.csv")

    # Save to spaCy format
    save_to_spacy(train_sentences, OUTPUT_DIR / "train.spacy")
    save_to_spacy(test_sentences, OUTPUT_DIR / "test.spacy")

    print("\nDone!")
    print(f"Total valid templates used: {total_templates}")


if __name__ == "__main__":
    main()
