"""NLP module for extracting departure and arrival cities."""

from .baseline import BaselineNER, NERResult
from .preprocessing import preprocess_text
from .spacy_ner import SpacyNER

# CamemBERT is optional (requires transformers/torch)
try:
    from .camembert_ner import CamemBERTNER, CamemBERTTrainer
    _CAMEMBERT_AVAILABLE = True
except ImportError:
    _CAMEMBERT_AVAILABLE = False
    CamemBERTNER = None
    CamemBERTTrainer = None

__all__ = [
    "BaselineNER",
    "SpacyNER",
    "CamemBERTNER",
    "CamemBERTTrainer",
    "NERResult",
    "preprocess_text",
]
