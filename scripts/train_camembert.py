"""
Training Script for CamemBERT NER Model

This script fine-tunes CamemBERT on the travel dataset for NER.

Usage:
    uv run python scripts/train_camembert.py
    uv run python scripts/train_camembert.py --epochs 5
    uv run python scripts/train_camembert.py --batch-size 8

Output:
    - Fine-tuned model saved to models/camembert-ner/
    - Training metrics saved to models/camembert-ner/metrics.json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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


def main():
    parser = argparse.ArgumentParser(description="Train CamemBERT NER model")
    parser.add_argument(
        "--train",
        type=str,
        default="data/train.spacy",
        help="Path to training data",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="data/test.spacy",
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/camembert-ner",
        help="Output directory for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="camembert-base",
        help="Base model name from HuggingFace",
    )

    args = parser.parse_args()

    # Check data exists
    if not check_data_exists():
        print("\nPlease run: uv run python scripts/make_dataset.py")
        sys.exit(1)

    # Paths
    train_path = PROJECT_ROOT / args.train
    eval_path = PROJECT_ROOT / args.eval
    output_dir = PROJECT_ROOT / args.output

    print("=" * 60)
    print("CamemBERT NER Training")
    print("=" * 60)
    print(f"Base model: {args.model_name}")
    print(f"Output: {output_dir}")
    print(f"Train data: {train_path}")
    print(f"Eval data: {eval_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    print()

    # Import and run training
    from src.nlp.camembert_ner import CamemBERTTrainer, TrainingConfig

    config = TrainingConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer = CamemBERTTrainer(config)
    results = trainer.train(train_path, eval_path, output_dir)

    print()
    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)
    print("\nFinal metrics:")
    for key, value in results.items():
        if key.startswith("eval_"):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
