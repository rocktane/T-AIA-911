"""
Evaluation Script for Travel Order Resolver NER Models

This script evaluates all NER models (Baseline, spaCy, CamemBERT) on the test dataset
and computes precision, recall, and F1-score for each entity type.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --model baseline
    uv run python scripts/evaluate.py --model spacy
    uv run python scripts/evaluate.py --model camembert
    uv run python scripts/evaluate.py --all

Output:
    - Metrics printed to console
    - JSON report saved to evaluation_report.json
    - Confusion matrices and examples of errors
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nlp.preprocessing import normalize_city_name


@dataclass
class EntityMetrics:
    """Metrics for a single entity type."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


@dataclass
class EvaluationResult:
    """Complete evaluation results for a model."""

    model_name: str
    departure_metrics: EntityMetrics
    arrival_metrics: EntityMetrics
    exact_match: float
    valid_detection_accuracy: float
    total_samples: int
    valid_samples: int
    invalid_samples: int
    errors: list  # Sample of errors for analysis


def load_test_data(csv_path: Path) -> list[dict]:
    """Load test data from CSV file."""
    samples = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": row["sentenceID"],
                "sentence": row["sentence"],
                "departure": row["departure"],
                "arrival": row["destination"],
                "is_valid": row["is_valid"].lower() == "true",
            })
    return samples


def load_cities(gares_path: Path) -> list[str]:
    """Load city names from gares CSV."""
    cities = []
    with open(gares_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append(row["nom"])
    return list(set(cities))


def normalize_for_comparison(text: str | None) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    return normalize_city_name(text)


def entity_matches(predicted: str | None, expected: str | None) -> bool:
    """Check if predicted entity matches expected."""
    if predicted is None and expected == "":
        return True
    if predicted is None or expected is None:
        return predicted == expected
    return normalize_for_comparison(predicted) == normalize_for_comparison(expected)


def evaluate_model(
    model,
    model_name: str,
    test_data: list[dict],
    max_errors: int = 20,
) -> EvaluationResult:
    """Evaluate a single model on test data."""
    dep_metrics = EntityMetrics()
    arr_metrics = EntityMetrics()
    exact_matches = 0
    valid_correct = 0
    errors = []

    for sample in test_data:
        sentence = sample["sentence"]
        expected_dep = sample["departure"]
        expected_arr = sample["arrival"]
        expected_valid = sample["is_valid"]

        # Get prediction
        result = model.extract(sentence)
        predicted_dep = result.departure
        predicted_arr = result.arrival
        predicted_valid = result.is_valid

        # Valid detection accuracy
        if predicted_valid == expected_valid:
            valid_correct += 1

        # Only evaluate entities for valid samples
        if expected_valid:
            # Departure
            if entity_matches(predicted_dep, expected_dep):
                dep_metrics.true_positives += 1
            else:
                if predicted_dep:
                    dep_metrics.false_positives += 1
                if expected_dep:
                    dep_metrics.false_negatives += 1

            # Arrival
            if entity_matches(predicted_arr, expected_arr):
                arr_metrics.true_positives += 1
            else:
                if predicted_arr:
                    arr_metrics.false_positives += 1
                if expected_arr:
                    arr_metrics.false_negatives += 1

            # Exact match (both correct)
            if entity_matches(predicted_dep, expected_dep) and entity_matches(predicted_arr, expected_arr):
                exact_matches += 1
            elif len(errors) < max_errors:
                errors.append({
                    "sentence": sentence,
                    "expected": {"departure": expected_dep, "arrival": expected_arr},
                    "predicted": {"departure": predicted_dep, "arrival": predicted_arr},
                })

    valid_samples = sum(1 for s in test_data if s["is_valid"])
    invalid_samples = len(test_data) - valid_samples

    return EvaluationResult(
        model_name=model_name,
        departure_metrics=dep_metrics,
        arrival_metrics=arr_metrics,
        exact_match=exact_matches / valid_samples if valid_samples > 0 else 0,
        valid_detection_accuracy=valid_correct / len(test_data) if test_data else 0,
        total_samples=len(test_data),
        valid_samples=valid_samples,
        invalid_samples=invalid_samples,
        errors=errors,
    )


def print_results(result: EvaluationResult) -> None:
    """Print evaluation results in a formatted table."""
    print(f"\n{'=' * 60}")
    print(f"Model: {result.model_name}")
    print(f"{'=' * 60}")
    print(f"Total samples: {result.total_samples}")
    print(f"  Valid: {result.valid_samples}")
    print(f"  Invalid: {result.invalid_samples}")
    print()

    # Metrics table
    print(f"{'Metric':<25} {'Departure':<15} {'Arrival':<15}")
    print("-" * 55)
    print(f"{'Precision':<25} {result.departure_metrics.precision:<15.4f} {result.arrival_metrics.precision:<15.4f}")
    print(f"{'Recall':<25} {result.departure_metrics.recall:<15.4f} {result.arrival_metrics.recall:<15.4f}")
    print(f"{'F1-Score':<25} {result.departure_metrics.f1:<15.4f} {result.arrival_metrics.f1:<15.4f}")
    print("-" * 55)
    print(f"{'Exact Match':<25} {result.exact_match:.4f}")
    print(f"{'Valid Detection Acc.':<25} {result.valid_detection_accuracy:.4f}")

    # Error analysis
    if result.errors:
        print(f"\n{'Sample Errors'}")
        print("-" * 55)
        for i, error in enumerate(result.errors[:5], 1):
            print(f"{i}. \"{error['sentence'][:60]}...\"")
            print(f"   Expected: dep={error['expected']['departure']}, arr={error['expected']['arrival']}")
            print(f"   Got:      dep={error['predicted']['departure']}, arr={error['predicted']['arrival']}")


def save_report(results: list[EvaluationResult], output_path: Path) -> None:
    """Save evaluation report to JSON."""
    report = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "models": [],
    }

    for result in results:
        model_report = {
            "name": result.model_name,
            "samples": {
                "total": result.total_samples,
                "valid": result.valid_samples,
                "invalid": result.invalid_samples,
            },
            "metrics": {
                "departure": {
                    "precision": result.departure_metrics.precision,
                    "recall": result.departure_metrics.recall,
                    "f1": result.departure_metrics.f1,
                    "true_positives": result.departure_metrics.true_positives,
                    "false_positives": result.departure_metrics.false_positives,
                    "false_negatives": result.departure_metrics.false_negatives,
                },
                "arrival": {
                    "precision": result.arrival_metrics.precision,
                    "recall": result.arrival_metrics.recall,
                    "f1": result.arrival_metrics.f1,
                    "true_positives": result.arrival_metrics.true_positives,
                    "false_positives": result.arrival_metrics.false_positives,
                    "false_negatives": result.arrival_metrics.false_negatives,
                },
                "exact_match": result.exact_match,
                "valid_detection_accuracy": result.valid_detection_accuracy,
            },
            "errors": result.errors,
        }
        report["models"].append(model_report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to {output_path}")


def generate_comparison_table(results: list[EvaluationResult]) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "| Metric | " + " | ".join(r.model_name for r in results) + " |",
        "|--------|" + "|".join("-" * 12 for _ in results) + "|",
    ]

    metrics = [
        ("Precision (Dep)", lambda r: r.departure_metrics.precision),
        ("Recall (Dep)", lambda r: r.departure_metrics.recall),
        ("F1 (Dep)", lambda r: r.departure_metrics.f1),
        ("Precision (Arr)", lambda r: r.arrival_metrics.precision),
        ("Recall (Arr)", lambda r: r.arrival_metrics.recall),
        ("F1 (Arr)", lambda r: r.arrival_metrics.f1),
        ("Exact Match", lambda r: r.exact_match),
        ("Valid Detection", lambda r: r.valid_detection_accuracy),
    ]

    for name, getter in metrics:
        values = [f"{getter(r):.4f}" for r in results]
        lines.append(f"| {name} | " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NER models")
    parser.add_argument(
        "--model",
        choices=["baseline", "spacy", "camembert", "all"],
        default="all",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/dataset_test.csv",
        help="Path to test data CSV",
    )
    parser.add_argument(
        "--gares",
        type=str,
        default="data/gares-france.csv",
        help="Path to gares CSV for city list",
    )
    parser.add_argument(
        "--camembert-model",
        type=str,
        default="models/camembert-ner",
        help="Path to fine-tuned CamemBERT model",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="models/spacy-ner/model-best",
        help="Path to trained spaCy model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_report.json",
        help="Output JSON report path",
    )

    args = parser.parse_args()

    # Paths
    test_path = PROJECT_ROOT / args.test_data
    gares_path = PROJECT_ROOT / args.gares
    output_path = PROJECT_ROOT / args.output

    # Check test data exists
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        print("Please run: uv run python scripts/make_dataset.py")
        sys.exit(1)

    # Load data
    print("Loading test data...")
    test_data = load_test_data(test_path)
    print(f"  Loaded {len(test_data)} samples")

    print("Loading city list...")
    cities = load_cities(gares_path)
    print(f"  Loaded {len(cities)} cities")

    results = []
    models_to_eval = ["baseline", "spacy", "camembert"] if args.model == "all" else [args.model]

    for model_name in models_to_eval:
        print(f"\nEvaluating {model_name}...")

        if model_name == "baseline":
            from src.nlp.baseline import BaselineNER
            model = BaselineNER(cities)
            result = evaluate_model(model, "Baseline", test_data)
            results.append(result)
            print_results(result)

        elif model_name == "spacy":
            from src.nlp.spacy_ner import SpacyNER

            # Try custom model first, fall back to pretrained
            spacy_model_path = PROJECT_ROOT / args.spacy_model
            if spacy_model_path.exists():
                print(f"  Using custom model: {spacy_model_path}")
                model = SpacyNER(cities, model_path=spacy_model_path)
                model_label = "spaCy (custom)"
            else:
                print("  Using pretrained model: fr_core_news_md")
                model = SpacyNER(cities, use_pretrained=True)
                model_label = "spaCy (pretrained)"

            result = evaluate_model(model, model_label, test_data)
            results.append(result)
            print_results(result)

        elif model_name == "camembert":
            camembert_path = PROJECT_ROOT / args.camembert_model
            if not camembert_path.exists():
                print(f"  Warning: CamemBERT model not found at {camembert_path}")
                print("  Please run: uv run python -m src.nlp.camembert_ner")
                continue

            from src.nlp.camembert_ner import CamemBERTNER
            model = CamemBERTNER(camembert_path, cities)
            result = evaluate_model(model, "CamemBERT", test_data)
            results.append(result)
            print_results(result)

    # Save report
    if results:
        save_report(results, output_path)

        # Print comparison table
        if len(results) > 1:
            print("\n" + "=" * 60)
            print("COMPARISON TABLE")
            print("=" * 60)
            print(generate_comparison_table(results))


if __name__ == "__main__":
    main()
