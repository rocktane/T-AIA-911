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
import unicodedata
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

# Configuration
SEED = 42
N_VALID_SENTENCES = 7000
N_INVALID_SENTENCES = 3000
TRAIN_RATIO = 0.8
DATA_DIR = Path(__file__).parent.parent / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
OUTPUT_DIR = DATA_DIR


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
        "invalid": load_templates(TEMPLATES_DIR / "invalid.txt"),
        "ambiguous": load_templates(TEMPLATES_DIR / "ambiguous.txt"),
        "name_city_contexts": load_templates(TEMPLATES_DIR / "name_city_contexts.txt"),
        "name_cities": load_templates(TEMPLATES_DIR / "name_cities.txt"),
        "compound_cities": load_templates(TEMPLATES_DIR / "compound_cities.txt"),
    }


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


def generate_valid_sentences(
    stations: list[str], n_samples: int, templates: dict
) -> list[dict]:
    """Generate valid travel order sentences."""
    sentences = []
    seen = set()
    all_templates = templates["dep_first"] + templates["arr_first"]
    name_cities = templates["name_cities"]
    name_city_contexts = templates["name_city_contexts"]
    compound_cities = templates.get("compound_cities", [])

    # Also use some name-city contexts
    name_cities_in_stations = [c for c in name_cities if c in stations]

    # Compound cities that exist in stations (exact match or prefix)
    compound_cities_in_stations = []
    for compound in compound_cities:
        # Check if compound city exists in stations (exact or with different formatting)
        compound_lower = compound.lower()
        for station in stations:
            station_lower = station.lower()
            # Match exact or station starts with compound name
            if station_lower == compound_lower or station_lower.startswith(compound_lower + " "):
                compound_cities_in_stations.append(station)
                break

    pbar = tqdm(total=n_samples, desc="Generating valid sentences")

    while len(sentences) < n_samples:
        rand_val = random.random()

        # 10% chance to use name-city context
        if rand_val < 0.10 and name_cities_in_stations and name_city_contexts:
            template = random.choice(name_city_contexts)
            name = random.choice(name_cities)
            dep = random.choice([s for s in stations if s != name])
            arr = random.choice([s for s in stations if s != dep and s != name])
            text = template.format(N=name, D=dep, A=arr)
        # 15% chance to use compound cities (important for training)
        elif rand_val < 0.25 and compound_cities_in_stations:
            template = random.choice(all_templates)
            # Use compound city for either departure or arrival
            if random.random() < 0.5:
                dep = random.choice(compound_cities_in_stations)
                arr = random.choice([s for s in stations if s != dep])
            else:
                dep = random.choice([s for s in stations])
                arr = random.choice([s for s in compound_cities_in_stations if s != dep])
            text = template.format(D=dep, A=arr)
        else:
            template = random.choice(all_templates)
            dep = random.choice(stations)
            arr = random.choice([s for s in stations if s != dep])
            text = template.format(D=dep, A=arr)

        # Apply noise (lowercase, no accents, typos)
        text, dep_noisy, arr_noisy = apply_noise(text, dep, arr)

        if text in seen:
            continue

        seen.add(text)
        sentences.append({
            "sentence": text,
            "departure": dep,
            "arrival": arr,
            "departure_noisy": dep_noisy,
            "arrival_noisy": arr_noisy,
            "is_valid": True,
        })
        pbar.update(1)

    pbar.close()
    return sentences


def generate_invalid_sentences(
    stations: list[str], n_samples: int, templates: dict
) -> list[dict]:
    """Generate invalid (non-travel) sentences."""
    sentences = []
    seen = set()
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
                "departure_noisy": "",
                "arrival_noisy": "",
                "is_valid": False,
            })
            pbar.update(1)

    # Ambiguous sentences - these have more variety with city names
    attempts = 0
    max_attempts = n_ambiguous * 10  # Prevent infinite loop
    while len(sentences) < n_samples and attempts < max_attempts:
        attempts += 1
        template = random.choice(ambiguous_templates)
        city = random.choice(stations)

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
            "departure_noisy": "",
            "arrival_noisy": "",
            "is_valid": False,
        })
        pbar.update(1)

    pbar.close()
    return sentences


def save_to_csv(
    sentences: list[dict], filepath: Path
) -> None:
    """Save sentences to CSV file."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentenceID", "sentence", "departure", "destination", "is_valid"])
        for i, s in enumerate(sentences, 1):
            writer.writerow([
                i,
                s["sentence"],
                s["departure"],
                s["arrival"],
                s["is_valid"],
            ])
    print(f"Saved {len(sentences)} sentences to {filepath}")


def save_to_spacy(
    sentences: list[dict], filepath: Path
) -> None:
    """Save sentences to spaCy DocBin format."""
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

        # Find entity positions using the noisy versions (what's actually in the text)
        dep_noisy = s["departure_noisy"]
        arr_noisy = s["arrival_noisy"]

        dep_start = text.find(dep_noisy)
        arr_start = text.find(arr_noisy)

        if dep_start == -1 or arr_start == -1:
            skipped += 1
            continue

        dep_end = dep_start + len(dep_noisy)
        arr_end = arr_start + len(arr_noisy)

        # Create spans
        span_dep = doc.char_span(dep_start, dep_end, label="DEPART")
        span_arr = doc.char_span(arr_start, arr_end, label="ARRIVEE")

        if span_dep is None or span_arr is None:
            skipped += 1
            continue

        # Check for overlap
        if not (dep_end <= arr_start or arr_end <= dep_start):
            skipped += 1
            continue

        doc.ents = [span_dep, span_arr]
        db.add(doc)

    db.to_disk(filepath)
    print(f"Saved {len(sentences) - skipped} sentences to {filepath} (skipped {skipped})")


def main():
    random.seed(SEED)

    # Load templates from files
    print("Loading templates...")
    templates = load_all_templates()

    total_templates = len(templates["dep_first"]) + len(templates["arr_first"])
    print(f"  Departure-first templates: {len(templates['dep_first'])}")
    print(f"  Arrival-first templates: {len(templates['arr_first'])}")
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

    # Generate sentences
    valid_sentences = generate_valid_sentences(stations, N_VALID_SENTENCES, templates)
    invalid_sentences = generate_invalid_sentences(stations, N_INVALID_SENTENCES, templates)

    # Combine and shuffle
    all_sentences = valid_sentences + invalid_sentences
    random.shuffle(all_sentences)

    print(f"\nTotal sentences: {len(all_sentences)}")
    print(f"  Valid: {len(valid_sentences)} ({len(valid_sentences)/len(all_sentences)*100:.1f}%)")
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
