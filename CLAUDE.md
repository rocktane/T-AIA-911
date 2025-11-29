# Travel Order Resolver

Projet NLP Epitech - Extraction de villes de depart/arrivee depuis des phrases en francais pour generer des itineraires ferroviaires SNCF.

## Stack Technique

- **Runtime**: Python 3.12+ avec `uv`
- **NLP**: spaCy (baseline) + CamemBERT (fine-tuning)
- **Geolocalisation**: scipy (KD-Tree), Haversine
- **Pathfinding**: NetworkX (Dijkstra)
- **Web**: FastAPI + Jinja2 (interface minimale)
- **Tests**: pytest
- **Linting**: ruff

## Commandes

```bash
# Installation des dependances
uv sync

# Lancer les tests
uv run pytest

# Lancer le linting
uv run ruff check .
uv run ruff format .

# Generer le dataset (10,000 phrases annotees)
uv run python scripts/make_dataset.py

# Entrainer le modele spaCy custom
uv run python scripts/train_spacy.py --max-steps 5000

# Entrainer le modele CamemBERT (necessite GPU recommande)
uv run python scripts/train_camembert.py --epochs 5

# Evaluer les modeles
uv run python scripts/evaluate.py --all
uv run python scripts/evaluate.py --model baseline
uv run python scripts/evaluate.py --model spacy
uv run python scripts/evaluate.py --model camembert

# Lancer l'interface web
uv run uvicorn src.web.app:app --reload

# Lancer le pipeline NLP (stdin)
cat input.csv | uv run python -m src.main

# Lancer le pipeline NLP (fichier)
uv run python -m src.main input.csv
```

## Pipeline

```
INPUT: sentenceID,sentence (UTF-8)
         |
         v
┌─────────────────────────────────────┐
│         MODULE NLP                  │
│  Preprocessing → NER → Classification│
│  (spaCy ou CamemBERT)               │
└─────────────────────────────────────┘
         |
    ┌────┴────┐
    v         v
 VALIDE    INVALIDE → sentenceID,INVALID
    |
    v
┌─────────────────────────────────────┐
│      MODULE GEOLOCALISATION         │
│  Ville = Gare ?                     │
│  NON → Gare la plus proche (KDTree) │
└─────────────────────────────────────┘
         |
         v
┌─────────────────────────────────────┐
│       MODULE PATHFINDING            │
│  Graphe SNCF → Dijkstra → Chemin    │
└─────────────────────────────────────┘
         |
         v
OUTPUT: sentenceID,ville_dep,gare_dep,ville_arr,gare_arr,itineraire
```

## Structure

```
T-AIA-911/
├── pyproject.toml              # Config uv + dependances
├── CLAUDE.md                   # Ce fichier
├── configs/
│   └── spacy_config.cfg        # Configuration entrainement spaCy
├── src/
│   ├── __init__.py
│   ├── main.py                 # Point d'entree CLI
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── preprocessing.py    # Nettoyage texte
│   │   ├── baseline.py         # Regex + dictionnaire
│   │   ├── spacy_ner.py        # spaCy NER
│   │   └── camembert_ner.py    # CamemBERT fine-tune
│   ├── geo/
│   │   ├── __init__.py
│   │   ├── distance.py         # Haversine
│   │   ├── nearest_station.py  # KD-Tree
│   │   └── city_resolver.py    # Ville → Gare
│   ├── pathfinding/
│   │   ├── __init__.py
│   │   ├── graph.py            # Construction graphe
│   │   └── dijkstra.py         # Recherche chemin
│   └── web/
│       ├── __init__.py
│       ├── app.py              # FastAPI
│       └── templates/
│           └── index.html      # Interface minimale
├── scripts/
│   ├── make_dataset.py         # Generation dataset
│   ├── train_spacy.py          # Entrainement spaCy
│   ├── train_camembert.py      # Entrainement CamemBERT
│   └── evaluate.py             # Calcul metriques
├── data/
│   ├── gares-france.csv        # Gares SNCF (nom,lon,lat,codeinsee)
│   ├── communes-france.csv     # Communes francaises
│   ├── connexions.csv          # Connexions entre gares
│   ├── templates/              # Templates de phrases
│   │   ├── valid.txt           # Phrases valides (depart/arrivee)
│   │   ├── invalid.txt         # Phrases invalides
│   │   └── ambiguous.txt       # Phrases ambigues
│   ├── train.spacy             # Dataset entrainement (8000 phrases)
│   └── test.spacy              # Dataset test (2000 phrases)
├── models/
│   ├── spacy-ner/              # Modele spaCy entraine
│   │   ├── model-best/         # Meilleur checkpoint
│   │   └── model-last/         # Dernier checkpoint
│   └── camembert-ner/          # Modele CamemBERT (optionnel)
└── tests/
    ├── test_nlp.py
    ├── test_geo.py
    └── test_pathfinding.py
```

## Formats I/O

### Entree (stdin ou fichier CSV)
```
sentenceID,sentence
1,Je veux aller de Paris a Lyon
2,Comment me rendre a Marseille depuis Bordeaux ?
3,Il fait beau aujourd'hui
```

### Sortie NLP
```
sentenceID,Departure,Destination
1,Paris,Lyon
2,Bordeaux,Marseille
3,INVALID
```

### Sortie Finale (avec geolocalisation + pathfinding)
```
sentenceID,ville_dep,gare_dep,ville_arr,gare_arr,itineraire
1,Paris,Paris Gare de Lyon,Lyon,Lyon Part-Dieu,"Paris→Lyon"
2,Bordeaux,Bordeaux Saint-Jean,Marseille,Marseille Saint-Charles,"Bordeaux→Toulouse→Marseille"
3,INVALID
```

## Donnees

### gares-france.csv
| Colonne | Description |
|---------|-------------|
| nom | Nom de la gare |
| libellecourt | Code court SNCF |
| lon | Longitude |
| lat | Latitude |
| codeinsee | Code INSEE |

### communes-france.csv
| Colonne | Description |
|---------|-------------|
| nom_commune | Nom de la commune |
| latitude | Latitude |
| longitude | Longitude |
| nom_departement | Departement |
| nom_region | Region |

## Metriques NLP

Evaluation sur 2000 phrases de test (1421 valides, 579 invalides).

| Metrique | Baseline | spaCy (custom) | CamemBERT |
|----------|----------|----------------|-----------|
| Precision (Dep) | 77.95% | 99.62% | **99.82%** |
| Recall (Dep) | 68.90% | 92.12% | **99.82%** |
| F1 (Dep) | 73.14% | 95.72% | **99.82%** |
| Precision (Arr) | 75.80% | 100.00% | **99.32%** |
| Recall (Arr) | 67.00% | 90.15% | **99.72%** |
| F1 (Arr) | 71.12% | 94.82% | **99.52%** |
| Exact Match | 63.48% | 85.22% | **~99%** |
| Valid Detection | 90.00% | 89.55% | - |

### Entrainement spaCy

- **Dataset** : 10,000 phrases (8000 train / 2000 test)
- **Architecture** : tok2vec + NER (TransitionBasedParser)
- **Steps** : 5000
- **F1 final** : 95.57%

### Entrainement CamemBERT

- **Modele de base** : `camembert-base`
- **Labels** : B-DEPART, I-DEPART, B-ARRIVEE, I-ARRIVEE, O
- **Hyperparametres** : epochs=5, batch_size=16, learning_rate=5e-5
- **Early stopping** : patience=3
- **F1 macro final** : **99.67%**
- **Eval loss** : 0.0099

## Cas Difficiles

Le modele gere :
- **Prenoms = villes** : Albert, Paris, Florence, Nancy, Lourdes
- **Noms composes** : Port-Boulet, Saint-Pierre-des-Corps, La Roche-sur-Yon
- **Sans majuscules** : `je veux aller de paris a lyon`
- **Sans accents** : `de beziers a montpellier`
- **Fautes d'orthographe** : Marseile → Marseille, Bordeau → Bordeaux, Toulous → Toulouse (fuzzy matching avec rapidfuzz)
- **Ordre inverse** : `A Marseille depuis Lyon` (arr avant dep)

## Ressources

- SNCF Open Data : https://ressources.data.sncf.com/
- spaCy francais : https://spacy.io/models/fr
- CamemBERT : https://huggingface.co/camembert-base
- CamemBERT NER : https://huggingface.co/Jean-Baptiste/camembert-ner
