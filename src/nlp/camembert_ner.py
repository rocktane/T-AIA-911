"""CamemBERT-based NER for extracting departure, arrival, and via cities.

This module implements fine-tuning of CamemBERT for Named Entity Recognition
specifically for travel-related entities (DEPART, ARRIVEE, and VIA).

Architecture:
- Base model: camembert-base from HuggingFace
- Task: Token classification (NER)
- Labels: O, B-DEPART, I-DEPART, B-ARRIVEE, I-ARRIVEE, B-VIA, I-VIA

Usage:
    # Training
    trainer = CamemBERTTrainer()
    trainer.train("data/train.spacy", "data/test.spacy", "models/camembert-ner")

    # Inference
    ner = CamemBERTNER("models/camembert-ner", cities=["Paris", "Lyon", ...])
    result = ner.extract("Je veux aller de Paris à Lyon")
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .baseline import NERResult, ViaPoint
from .preprocessing import normalize_city_name


# Label mapping for NER
LABEL2ID = {
    "O": 0,
    "B-DEPART": 1,
    "I-DEPART": 2,
    "B-ARRIVEE": 3,
    "I-ARRIVEE": 4,
    "B-VIA": 5,
    "I-VIA": 6,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)


@dataclass
class TrainingConfig:
    """Configuration for CamemBERT training."""

    model_name: str = "camembert-base"
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 50
    early_stopping_patience: int = 3
    fp16: bool = torch.cuda.is_available()


class NERDataset(Dataset):
    """Dataset for NER training from spaCy DocBin format."""

    def __init__(
        self,
        spacy_path: str | Path,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        """
        Initialize dataset from spaCy DocBin file.

        Args:
            spacy_path: Path to .spacy file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        import spacy
        from spacy.tokens import DocBin

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load spaCy docs
        nlp = spacy.blank("fr")
        doc_bin = DocBin().from_disk(spacy_path)
        docs = list(doc_bin.get_docs(nlp.vocab))

        # Convert to training examples
        for doc in docs:
            text = doc.text
            entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            self.examples.append({"text": text, "entities": entities})

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        text = example["text"]
        entities = example["entities"]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        # Get offset mapping for aligning labels
        offset_mapping = encoding["offset_mapping"].squeeze().tolist()

        # Create labels aligned with tokens
        labels = []
        for offset in offset_mapping:
            if offset == (0, 0):
                # Special token
                labels.append(-100)
            else:
                start, end = offset
                label = "O"

                # Check if this token overlaps with any entity
                for ent_start, ent_end, ent_label in entities:
                    if start >= ent_start and end <= ent_end:
                        # Token is inside entity
                        if start == ent_start:
                            label = f"B-{ent_label}"
                        else:
                            label = f"I-{ent_label}"
                        break

                labels.append(LABEL2ID.get(label, 0))

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(labels),
        }


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute NER metrics (precision, recall, F1) for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Compute metrics per entity type
    metrics = {}
    for entity in ["DEPART", "ARRIVEE"]:
        b_label = LABEL2ID[f"B-{entity}"]
        i_label = LABEL2ID[f"I-{entity}"]

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_seq, label_seq in zip(predictions, labels):
            pred_entities = _extract_entities(pred_seq, b_label, i_label)
            true_entities = _extract_entities(label_seq, b_label, i_label)

            for pe in pred_entities:
                if pe in true_entities:
                    true_positives += 1
                else:
                    false_positives += 1

            for te in true_entities:
                if te not in pred_entities:
                    false_negatives += 1

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[f"precision_{entity.lower()}"] = precision
        metrics[f"recall_{entity.lower()}"] = recall
        metrics[f"f1_{entity.lower()}"] = f1

    # Overall metrics
    metrics["f1_macro"] = (metrics["f1_depart"] + metrics["f1_arrivee"]) / 2

    return metrics


def _extract_entities(sequence: np.ndarray, b_label: int, i_label: int) -> list[tuple[int, int]]:
    """Extract entity spans from a label sequence."""
    entities = []
    start = None

    for i, label in enumerate(sequence):
        if label == -100:
            continue
        if label == b_label:
            if start is not None:
                entities.append((start, i))
            start = i
        elif label == i_label:
            if start is None:
                start = i
        else:
            if start is not None:
                entities.append((start, i))
                start = None

    if start is not None:
        entities.append((start, len(sequence)))

    return entities


class CamemBERTTrainer:
    """Trainer for fine-tuning CamemBERT on NER task."""

    def __init__(self, config: TrainingConfig | None = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.tokenizer = None
        self.model = None

    def train(
        self,
        train_path: str | Path,
        eval_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, float]:
        """
        Fine-tune CamemBERT on NER dataset.

        Args:
            train_path: Path to training .spacy file
            eval_path: Path to evaluation .spacy file
            output_dir: Directory to save the model

        Returns:
            Dictionary with final metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        print(f"Loading tokenizer from {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model
        print(f"Loading model from {self.config.model_name}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # Create datasets
        print("Loading training dataset...")
        train_dataset = NERDataset(train_path, self.tokenizer, self.config.max_length)
        print(f"  Loaded {len(train_dataset)} training examples")

        print("Loading evaluation dataset...")
        eval_dataset = NERDataset(eval_path, self.tokenizer, self.config.max_length)
        print(f"  Loaded {len(eval_dataset)} evaluation examples")

        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            fp16=self.config.fp16,
            report_to="none",  # Disable wandb/tensorboard
            save_total_limit=2,
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )

        # Train
        print("\nStarting training...")
        train_result = trainer.train()

        # Save final model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))

        # Save config
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        # Final evaluation
        print("\nFinal evaluation...")
        eval_results = trainer.evaluate()

        # Save metrics
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        print("\nTraining complete!")
        print(f"  F1 (DEPART): {eval_results.get('eval_f1_depart', 0):.4f}")
        print(f"  F1 (ARRIVEE): {eval_results.get('eval_f1_arrivee', 0):.4f}")
        print(f"  F1 (macro): {eval_results.get('eval_f1_macro', 0):.4f}")

        return eval_results


class CamemBERTNER:
    """CamemBERT-based NER for inference."""

    def __init__(
        self,
        model_path: str | Path,
        cities: list[str] | None = None,
        communes: list[str] | None = None,
        device: str | None = None,
    ):
        """
        Initialize CamemBERT NER for inference.

        Args:
            model_path: Path to fine-tuned model
            cities: List of known city names for validation (stations)
            communes: List of commune names for extended matching
            device: Device to use (cuda/cpu/mps)
        """
        self.model_path = Path(model_path)

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Load model and tokenizer
        print(f"Loading CamemBERT NER from {model_path} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        # City validation (stations)
        self.cities = set(cities) if cities else set()
        self.cities_normalized = {normalize_city_name(c): c for c in self.cities}

        # Commune validation (for extended coverage)
        self.communes = set(communes) if communes else set()
        self.communes_normalized = {normalize_city_name(c): c for c in self.communes}

    def _validate_city(self, text: str) -> str | None:
        """Check if text is a known city and return canonical name.

        Priority order:
        1. Exact match in stations
        2. Prefix match in stations
        3. Exact match in communes
        4. Prefix match in communes
        """
        if not self.cities and not self.communes:
            return text  # No validation if no city/commune list

        text_norm = normalize_city_name(text)

        # Priority 1: Exact match in stations
        if text_norm in self.cities_normalized:
            return self.cities_normalized[text_norm]

        # Priority 2: Prefix match in stations
        for city_norm, city_original in self.cities_normalized.items():
            if city_norm.startswith(text_norm + " ") or city_norm == text_norm:
                return city_original

        # Priority 3: Exact match in communes
        if text_norm in self.communes_normalized:
            return self.communes_normalized[text_norm]

        # Priority 4: Prefix match in communes
        for commune_norm, commune_original in self.communes_normalized.items():
            if commune_norm.startswith(text_norm + " ") or commune_norm == text_norm:
                return commune_original

        return None

    def extract(self, sentence: str) -> NERResult:
        """
        Extract departure, arrival, and VIA waypoints from a sentence.

        Args:
            sentence: Input sentence

        Returns:
            NERResult with departure, arrival, vias, and validity
        """
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=128,
        )

        offset_mapping = encoding["offset_mapping"].squeeze().tolist()

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2).squeeze().cpu().numpy()
            # Get confidence scores
            probs = torch.softmax(outputs.logits, dim=2).squeeze().cpu().numpy()

        # Extract entities from predictions
        departure = None
        arrival = None
        vias = []
        dep_start, dep_end = None, None
        arr_start, arr_end = None, None
        dep_confidence, arr_confidence = 0.0, 0.0

        current_entity = None
        current_start = None
        current_tokens = []
        current_probs = []

        def save_entity(entity_type, entity_text, start, end, avg_prob):
            """Helper to save an extracted entity."""
            nonlocal departure, arrival, dep_start, dep_end, arr_start, arr_end
            nonlocal dep_confidence, arr_confidence, vias

            validated = self._validate_city(entity_text)
            if not validated:
                return

            if entity_type == "DEPART":
                departure = validated
                dep_start = start
                dep_end = end
                dep_confidence = avg_prob
            elif entity_type == "ARRIVEE":
                arrival = validated
                arr_start = start
                arr_end = end
                arr_confidence = avg_prob
            elif entity_type == "VIA":
                vias.append(ViaPoint(
                    city=validated,
                    start=start,
                    end=end,
                    confidence=avg_prob,
                    order=start,
                ))

        for i, (pred, offset, prob) in enumerate(zip(predictions, offset_mapping, probs)):
            if offset == [0, 0]:
                continue

            label = ID2LABEL.get(pred, "O")

            if label.startswith("B-"):
                # Save previous entity
                if current_entity and current_tokens:
                    prev_end = offset_mapping[i-1][1] if i > 0 else offset[1]
                    entity_text = self._reconstruct_entity(sentence, current_start, prev_end)
                    avg_prob = np.mean(current_probs)
                    save_entity(current_entity, entity_text, current_start, prev_end, avg_prob)

                # Start new entity
                current_entity = label[2:]  # Remove "B-"
                current_start = offset[0]
                current_tokens = [i]
                current_probs = [prob[pred]]

            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue entity
                current_tokens.append(i)
                current_probs.append(prob[pred])

            else:
                # End of entity
                if current_entity and current_tokens:
                    last_token_idx = current_tokens[-1]
                    end_pos = offset_mapping[last_token_idx][1]
                    entity_text = self._reconstruct_entity(sentence, current_start, end_pos)
                    avg_prob = np.mean(current_probs)
                    save_entity(current_entity, entity_text, current_start, end_pos, avg_prob)

                current_entity = None
                current_start = None
                current_tokens = []
                current_probs = []

        # Handle last entity
        if current_entity and current_tokens:
            last_token_idx = current_tokens[-1]
            end_pos = offset_mapping[last_token_idx][1]
            entity_text = self._reconstruct_entity(sentence, current_start, end_pos)
            avg_prob = np.mean(current_probs)
            save_entity(current_entity, entity_text, current_start, end_pos, avg_prob)

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

    def _reconstruct_entity(self, text: str, start: int, end: int) -> str:
        """Reconstruct entity text from character positions."""
        return text[start:end].strip()

    def process_batch(self, sentences: list[str]) -> list[NERResult]:
        """Process multiple sentences."""
        return [self.extract(s) for s in sentences]


def main():
    """CLI for training CamemBERT NER."""
    import argparse

    parser = argparse.ArgumentParser(description="Train CamemBERT for NER")
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

    args = parser.parse_args()

    config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer = CamemBERTTrainer(config)
    trainer.train(args.train, args.eval, args.output)


if __name__ == "__main__":
    main()
