# Travel Order Resolver

NLP system to extract departure/arrival cities from French sentences and generate SNCF train itineraries.

## Overview

This project uses Natural Language Processing to:
1. Extract departure and arrival cities from French travel-related sentences
2. Resolve cities to their nearest SNCF train stations
3. Generate optimal train itineraries using pathfinding algorithms

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

```bash
# Install dependencies
uv sync

# Download spaCy French language model
uv run python -m spacy download fr_core_news_md

# Generate dataset (10,000 annotated sentences)
uv run python scripts/make_dataset.py

# Train custom spaCy NER model
uv run python scripts/train_spacy.py --max-steps 5000

# (Optional) Train CamemBERT model (requires GPU)
uv run python scripts/train_camembert.py --epochs 5
```

## Usage

### Command Line Interface

```bash
# From stdin
echo "1,Je veux aller de Paris à Lyon" | uv run python -m src.main

# From file
uv run python -m src.main input.csv
```

### Web Interface

```bash
# Launch FastAPI web server
uv run uvicorn src.web.app:app --reload
```

Then open http://localhost:8000 in your browser.

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov
```

## Development

```bash
# Code linting
uv run ruff check .

# Code formatting
uv run ruff format .
```

## Architecture

The project follows a three-module pipeline:

1. **NLP Module**: Extracts departure and arrival cities from sentences
   - Preprocessing (lowercase, accent normalization)
   - Named Entity Recognition (spaCy or CamemBERT)
   - Classification (departure/arrival)

2. **Geolocation Module**: Resolves cities to SNCF train stations
   - KD-Tree for nearest station search
   - Haversine distance calculation

3. **Pathfinding Module**: Computes optimal train routes
   - NetworkX graph representation
   - Dijkstra algorithm for shortest path

## Input/Output Format

### Input (CSV)
```csv
sentenceID,sentence
1,Je veux aller de Paris à Lyon
2,Comment me rendre à Marseille depuis Bordeaux ?
3,Il fait beau aujourd'hui
```

### Output (CSV)
```csv
sentenceID,ville_dep,gare_dep,ville_arr,gare_arr,itineraire
1,Paris,Paris Gare de Lyon,Lyon,Lyon Part-Dieu,"Paris→Lyon"
2,Bordeaux,Bordeaux Saint-Jean,Marseille,Marseille Saint-Charles,"Bordeaux→Toulouse→Marseille"
3,INVALID
```

## NLP Metrics

Evaluation on 2,000 test sentences (1,421 valid, 579 invalid):

| Metric | Baseline | spaCy (custom) |
|--------|----------|----------------|
| Precision (Dep) | 77.95% | **99.62%** |
| Recall (Dep) | 68.90% | **92.12%** |
| F1 (Dep) | 73.14% | **95.72%** |
| Precision (Arr) | 75.80% | **100.00%** |
| Recall (Arr) | 67.00% | **90.15%** |
| F1 (Arr) | 71.12% | **94.82%** |
| Exact Match | 63.48% | **85.22%** |

Evaluate models:
```bash
uv run python scripts/evaluate.py --all
```

## Project Structure

```
T-AIA-911/
├── src/
│   ├── main.py              # CLI entry point
│   ├── nlp/                 # NLP extraction
│   ├── geo/                 # Geolocation
│   ├── pathfinding/         # Route computation
│   └── web/                 # Web interface
├── scripts/
│   ├── make_dataset.py      # Dataset generation
│   ├── train_spacy.py       # spaCy training
│   ├── train_camembert.py   # CamemBERT training
│   └── evaluate.py          # Model evaluation
├── configs/
│   └── spacy_config.cfg     # spaCy training config
├── data/
│   ├── gares-france.csv     # SNCF stations
│   ├── communes-france.csv  # French cities
│   ├── connexions.csv       # Station connections
│   └── templates/           # Sentence templates
├── models/
│   └── spacy-ner/           # Trained models
└── tests/                   # Unit tests
```

## Technologies

- **NLP**: spaCy, CamemBERT (Transformers)
- **Geolocation**: scipy (KD-Tree), Haversine formula
- **Pathfinding**: NetworkX (Dijkstra)
- **Web**: FastAPI, Uvicorn, Jinja2
- **Testing**: pytest
- **Linting**: ruff

## License

Epitech project
