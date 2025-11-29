"""
Training Script for spaCy NER Model

This script trains a custom spaCy NER model on the travel dataset.

Usage:
    uv run python scripts/train_spacy.py
    uv run python scripts/train_spacy.py --epochs 30
    uv run python scripts/train_spacy.py --gpu

Output:
    - Trained model saved to models/spacy-ner/
    - Training logs and metrics
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_data_exists() -> bool:
    """Check if training data exists."""
    train_path = PROJECT_ROOT / "data" / "train.spacy"
    test_path = PROJECT_ROOT / "data" / "test.spacy"

    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        return False
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return False

    return True


def run_spacy_train(
    config_path: Path,
    output_dir: Path,
    train_path: Path,
    dev_path: Path,
    gpu: bool = False,
    max_steps: int = 20000,
) -> int:
    """Run spaCy training command."""
    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(train_path),
        "--paths.dev", str(dev_path),
        "--training.max_steps", str(max_steps),
    ]

    if gpu:
        cmd.extend(["--gpu-id", "0"])

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train spaCy NER model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/spacy_config.cfg",
        help="Path to spaCy config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/spacy-ner",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/train.spacy",
        help="Path to training data",
    )
    parser.add_argument(
        "--dev",
        type=str,
        default="data/test.spacy",
        help="Path to validation/dev data",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for training",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Maximum training steps",
    )

    args = parser.parse_args()

    # Paths
    config_path = PROJECT_ROOT / args.config
    output_dir = PROJECT_ROOT / args.output
    train_path = PROJECT_ROOT / args.train
    dev_path = PROJECT_ROOT / args.dev

    # Check config exists
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    # Check data exists
    if not check_data_exists():
        print("\nPlease run: uv run python scripts/make_dataset.py")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("spaCy NER Training")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Train data: {train_path}")
    print(f"Dev data: {dev_path}")
    print(f"GPU: {args.gpu}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 60)
    print()

    # Run training
    returncode = run_spacy_train(
        config_path=config_path,
        output_dir=output_dir,
        train_path=train_path,
        dev_path=dev_path,
        gpu=args.gpu,
        max_steps=args.max_steps,
    )

    if returncode == 0:
        print()
        print("=" * 60)
        print("Training complete!")
        print(f"Best model saved to: {output_dir}/model-best")
        print(f"Last model saved to: {output_dir}/model-last")
        print("=" * 60)

        # Copy best model to main location
        best_model = output_dir / "model-best"
        if best_model.exists():
            print(f"\nTo use the trained model:")
            print(f"  model = SpacyNER(cities, model_path='{best_model}')")
    else:
        print(f"\nTraining failed with return code: {returncode}")
        sys.exit(returncode)


if __name__ == "__main__":
    main()
