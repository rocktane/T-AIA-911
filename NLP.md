# NLP - Travel Order Resolver

## Objectif

Extraire la ville de depart (DEP) et la ville d'arrivee (ARR) d'une phrase en francais.

---

## Approche Recommandee (Progressive)

### Niveau 1 : Baseline (Regex + Dictionnaire)

**Principe** : Utiliser des regles simples pour detecter les villes et leur role.

```python
# src/nlp/baseline.py

import re

class BaselineNER:
    def __init__(self, cities_list):
        self.cities = set(c.lower() for c in cities_list)

        # Patterns pour identifier le role
        self.dep_patterns = [
            r"(?:de|depuis|partant de|en partance de)\s+(\w+)",
            r"(\w+)\s*[-–]\s*\w+",  # Paris-Lyon
        ]
        self.arr_patterns = [
            r"(?:a|vers|pour|direction)\s+(\w+)",
            r"\w+\s*[-–]\s*(\w+)",  # Paris-Lyon
        ]

    def extract(self, sentence):
        sentence_lower = sentence.lower()

        # Trouver toutes les villes mentionnees
        found_cities = []
        for city in self.cities:
            if city in sentence_lower:
                found_cities.append(city)

        if len(found_cities) < 2:
            return None, None

        # Appliquer les patterns
        departure = None
        destination = None

        for pattern in self.dep_patterns:
            match = re.search(pattern, sentence_lower)
            if match and match.group(1) in self.cities:
                departure = match.group(1)
                break

        for pattern in self.arr_patterns:
            match = re.search(pattern, sentence_lower)
            if match and match.group(1) in self.cities:
                destination = match.group(1)
                break

        return departure, destination
```

**Avantages** :
- Simple a implementer
- Rapide
- Facile a debugger

**Inconvenients** :
- Ne gere pas les cas ambigus
- Fragile aux variations de formulation

---

### Niveau 2 : spaCy (NER pre-entraine)

**Principe** : Utiliser le NER de spaCy pour detecter les entites LOC, puis classifier.

```python
# src/nlp/spacy_ner.py

import spacy

class SpacyNER:
    def __init__(self, cities_list):
        self.nlp = spacy.load("fr_core_news_md")
        self.cities = set(c.lower() for c in cities_list)

    def extract(self, sentence):
        doc = self.nlp(sentence)

        # Extraire les entites de type LOC
        locations = []
        for ent in doc.ents:
            if ent.label_ == "LOC":
                locations.append({
                    "text": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                })

        # Filtrer par notre liste de villes
        valid_locations = [
            loc for loc in locations
            if loc["text"].lower() in self.cities
        ]

        if len(valid_locations) < 2:
            return None, None

        # Classifier depart/arrivee par position et contexte
        departure, destination = self._classify(sentence, valid_locations)

        return departure, destination

    def _classify(self, sentence, locations):
        # Logique de classification basee sur le contexte
        # (similaire a la baseline mais avec les positions exactes)
        pass
```

**Installation** :
```bash
pip install spacy
python -m spacy download fr_core_news_md
```

---

### Niveau 3 : CamemBERT (Fine-tuning)

**Principe** : Fine-tuner un modele BERT francais pour notre tache specifique de NER.

```python
# src/nlp/camembert.py

from transformers import CamembertTokenizer, CamembertForTokenClassification
import torch

class CamembertNER:
    def __init__(self, model_path):
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model = CamembertForTokenClassification.from_pretrained(model_path)
        self.labels = ["O", "B-DEP", "I-DEP", "B-ARR", "I-ARR"]

    def extract(self, sentence):
        # Tokenization
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True
        )

        # Prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)

        # Decoder les labels
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.labels[p] for p in predictions[0]]

        # Extraire DEP et ARR
        departure = self._extract_entity(tokens, labels, "DEP")
        destination = self._extract_entity(tokens, labels, "ARR")

        return departure, destination

    def _extract_entity(self, tokens, labels, entity_type):
        entity_tokens = []
        for token, label in zip(tokens, labels):
            if entity_type in label:
                entity_tokens.append(token)

        if entity_tokens:
            return self.tokenizer.convert_tokens_to_string(entity_tokens)
        return None
```

**Ressource** : https://huggingface.co/camembert-base

---

## Metriques d'Evaluation

### Precision, Recall, F1

```python
# src/utils/metrics.py

def compute_metrics(predictions, ground_truth):
    """
    predictions: list of (dep, arr) tuples
    ground_truth: list of (dep, arr) tuples
    """
    tp_dep, fp_dep, fn_dep = 0, 0, 0
    tp_arr, fp_arr, fn_arr = 0, 0, 0

    for pred, truth in zip(predictions, ground_truth):
        # Departure
        if pred[0] == truth[0]:
            tp_dep += 1
        else:
            if pred[0] is not None:
                fp_dep += 1
            if truth[0] is not None:
                fn_dep += 1

        # Destination
        if pred[1] == truth[1]:
            tp_arr += 1
        else:
            if pred[1] is not None:
                fp_arr += 1
            if truth[1] is not None:
                fn_arr += 1

    # Calcul des metriques
    precision_dep = tp_dep / (tp_dep + fp_dep) if (tp_dep + fp_dep) > 0 else 0
    recall_dep = tp_dep / (tp_dep + fn_dep) if (tp_dep + fn_dep) > 0 else 0
    f1_dep = 2 * precision_dep * recall_dep / (precision_dep + recall_dep) if (precision_dep + recall_dep) > 0 else 0

    precision_arr = tp_arr / (tp_arr + fp_arr) if (tp_arr + fp_arr) > 0 else 0
    recall_arr = tp_arr / (tp_arr + fn_arr) if (tp_arr + fn_arr) > 0 else 0
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr) if (precision_arr + recall_arr) > 0 else 0

    return {
        "departure": {"precision": precision_dep, "recall": recall_dep, "f1": f1_dep},
        "destination": {"precision": precision_arr, "recall": recall_arr, "f1": f1_arr}
    }
```

### Exact Match

```python
def exact_match(predictions, ground_truth):
    """Pourcentage de predictions parfaitement correctes"""
    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    return correct / len(predictions)
```

---

## Patterns Linguistiques Francais

### Indicateurs de Depart
| Pattern | Exemple |
|---------|---------|
| `de [VILLE]` | de Paris |
| `depuis [VILLE]` | depuis Lyon |
| `partant de [VILLE]` | partant de Bordeaux |
| `en partance de [VILLE]` | en partance de Marseille |
| `au depart de [VILLE]` | au depart de Lille |

### Indicateurs d'Arrivee
| Pattern | Exemple |
|---------|---------|
| `a [VILLE]` | a Lyon |
| `vers [VILLE]` | vers Paris |
| `pour [VILLE]` | pour Marseille |
| `direction [VILLE]` | direction Bordeaux |
| `jusqu'a [VILLE]` | jusqu'a Nice |

### Cas Speciaux
| Pattern | Interpretation |
|---------|----------------|
| `[VILLE1] [VILLE2]` | DEP: VILLE1, ARR: VILLE2 |
| `[VILLE1]-[VILLE2]` | DEP: VILLE1, ARR: VILLE2 |
| `[VILLE1] vers [VILLE2]` | DEP: VILLE1, ARR: VILLE2 |

---

## Entrainement des Modeles

### Generation du Dataset

Le dataset est genere automatiquement a partir de templates et de la liste des gares SNCF :

```bash
uv run python scripts/make_dataset.py
```

**Output** :
- `data/train.spacy` : 8000 phrases d'entrainement
- `data/test.spacy` : 2000 phrases de test
- `data/dataset_train.csv` : Format CSV pour analyse
- `data/dataset_test.csv` : Format CSV pour evaluation

### Entrainement spaCy

```bash
uv run python scripts/train_spacy.py --max-steps 5000
```

**Configuration** (`configs/spacy_config.cfg`) :
- Architecture : tok2vec + NER (TransitionBasedParser)
- Embeddings : MultiHashEmbed (96 dimensions)
- Encoder : MaxoutWindowEncoder (depth=4)
- Optimiseur : Adam avec warmup lineaire

**Resultats** :
| Metrique | Score |
|----------|-------|
| F1 (global) | 95.57% |
| Precision | 96.72% |
| Recall | 94.44% |

### Entrainement CamemBERT

```bash
uv run python scripts/train_camembert.py --epochs 5 --batch-size 16
```

**Configuration** :
- Modele de base : `camembert-base`
- Labels BIO : O, B-DEPART, I-DEPART, B-ARRIVEE, I-ARRIVEE
- Learning rate : 5e-5
- Warmup : 10% des steps

### Evaluation

```bash
# Tous les modeles
uv run python scripts/evaluate.py --all

# Modele specifique
uv run python scripts/evaluate.py --model baseline
uv run python scripts/evaluate.py --model spacy
uv run python scripts/evaluate.py --model camembert
```

**Resultats d'Evaluation** :

| Metrique | Baseline | spaCy (custom) | CamemBERT |
|----------|----------|----------------|-----------|
| Precision (Dep) | 77.95% | **99.62%** | - |
| Recall (Dep) | 68.90% | **92.12%** | - |
| F1 (Dep) | 73.14% | **95.72%** | - |
| Precision (Arr) | 75.80% | **100.00%** | - |
| Recall (Arr) | 67.00% | **90.15%** | - |
| F1 (Arr) | 71.12% | **94.82%** | - |
| Exact Match | 63.48% | **85.22%** | - |

---

## Ressources

- **spaCy francais** : https://spacy.io/models/fr
- **CamemBERT** : https://huggingface.co/camembert-base
- **CamemBERT NER** : https://huggingface.co/Jean-Baptiste/camembert-ner
- **Tutorial NER HuggingFace** : https://huggingface.co/docs/transformers/tasks/token_classification
