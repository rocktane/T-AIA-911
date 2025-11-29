# Dataset - Travel Order Resolver

## Objectif

Creer un dataset annote pour entrainer et evaluer le modele NER.

**Cible** : ~10 000 phrases, ~300 structures grammaticales differentes

---

## Format d'Annotation

### Format pour l'entrainement
```
Je veux aller de <Dep>Paris</Dep> a <Arr>Lyon</Arr>.
Depuis <Dep>Bordeaux</Dep>, je souhaite me rendre a <Arr>Marseille</Arr>.
```

### Format IOB (pour spaCy/CamemBERT)
```
Je        O
veux      O
aller     O
de        O
Paris     B-DEP
a         O
Lyon      B-ARR
.         O
```

### Format CSV final
```csv
sentenceID,sentence,departure,destination,is_valid
1,"Je veux aller de Paris a Lyon",Paris,Lyon,true
2,"Il fait beau",,,false
```

---

## Categories de Phrases

### 1. Phrases Valides (~70% du dataset)

#### Structures de base
```
- Je veux aller de [DEP] a [ARR]
- Je souhaite me rendre a [ARR] depuis [DEP]
- Comment aller de [DEP] vers [ARR] ?
- Un billet [DEP] [ARR] s'il vous plait
- [DEP] - [ARR]
- De [DEP] pour [ARR]
- En partance de [DEP] vers [ARR]
- A quelle heure y a-t-il des trains de [DEP] a [ARR] ?
```

#### Variations de formulation
```
- Je voudrais partir de [DEP] pour aller a [ARR]
- Depuis [DEP], direction [ARR]
- Train [DEP] [ARR]
- [DEP] vers [ARR], c'est possible ?
```

### 2. Phrases Invalides (~30% du dataset)

#### Hors-sujet
```
- Il fait beau aujourd'hui
- Quelle heure est-il ?
- Je voudrais un cafe
- Bonjour, comment allez-vous ?
```

#### Ambigues (pas assez d'info)
```
- Je veux aller a Paris (manque depart)
- Depuis Lyon (manque arrivee)
- Un billet s'il vous plait (manque tout)
```

---

## Cas Difficiles a Inclure

### Prenoms = Noms de villes
```
- Avec mon ami Albert, je veux aller de Paris a Lyon
  → DEP: Paris, ARR: Lyon (Albert est un prenom)

- Je veux aller a Albert depuis Paris
  → DEP: Paris, ARR: Albert (Albert est une ville)
```

**Villes-prenoms** : Albert, Paris, Florence, Nancy, Lourdes, Nice

### Villes avec mots communs
```
- Port-Boulet ("port" et "boulet" sont des noms communs)
- La Roche-sur-Yon
- Saint-Pierre-des-Corps
```

### Ordre inverse (destination avant depart)
```
- A Marseille, j'aimerais y aller depuis Lyon
  → DEP: Lyon, ARR: Marseille
```

### Sans majuscules/accents
```
- je veux aller de paris a lyon
- de montpellier vers beziers
```

### Fautes d'orthographe
```
- Je veux aller a Marseile (Marseille)
- De Bordeau a Lyon (Bordeaux)
- Vers Toulous depuis Paris (Toulouse)
```

---

## Processus de Creation

### Etape 1 : Lister les villes SNCF
```python
# Telecharger depuis data.sncf.com
# Extraire ~300 villes principales
villes = ["Paris", "Lyon", "Marseille", "Bordeaux", ...]
```

### Etape 2 : Definir les templates
```python
templates = [
    "Je veux aller de {dep} a {arr}",
    "Depuis {dep}, direction {arr}",
    "Un billet {dep} {arr}",
    # ... ~50-100 templates
]
```

### Etape 3 : Generer les phrases
```python
import random

def generate_sentences(templates, villes, n=10000):
    sentences = []
    for _ in range(n):
        template = random.choice(templates)
        dep = random.choice(villes)
        arr = random.choice([v for v in villes if v != dep])
        sentence = template.format(dep=dep, arr=arr)
        sentences.append({
            "sentence": sentence,
            "departure": dep,
            "destination": arr,
            "is_valid": True
        })
    return sentences
```

### Etape 4 : Ajouter les perturbations
```python
def add_noise(sentence):
    # Sans majuscules
    if random.random() < 0.2:
        sentence = sentence.lower()
    # Fautes d'orthographe
    if random.random() < 0.1:
        sentence = add_typos(sentence)
    return sentence
```

### Etape 5 : Ajouter les phrases invalides
```python
invalid_sentences = [
    "Bonjour",
    "Il fait beau",
    "Quelle heure est-il ?",
    # ... ~3000 phrases invalides
]
```

---

## Script de Generation

```python
# scripts/generate_dataset.py

import pandas as pd
import random

# Charger les villes
villes = pd.read_csv("data/gares_sncf.csv")["ville"].tolist()

# Templates
templates_valid = [
    ("Je veux aller de {dep} a {arr}", "dep_first"),
    ("Depuis {dep}, direction {arr}", "dep_first"),
    ("A {arr} depuis {dep}", "arr_first"),
    # ...
]

templates_invalid = [
    "Bonjour",
    "Il fait beau aujourd'hui",
    "Je voudrais un cafe",
    # ...
]

def generate_dataset(n_valid=7000, n_invalid=3000):
    data = []

    # Phrases valides
    for i in range(n_valid):
        template, _ = random.choice(templates_valid)
        dep = random.choice(villes)
        arr = random.choice([v for v in villes if v != dep])
        sentence = template.format(dep=dep, arr=arr)
        data.append({
            "id": i,
            "sentence": sentence,
            "departure": dep,
            "destination": arr,
            "is_valid": True
        })

    # Phrases invalides
    for i, sentence in enumerate(random.choices(templates_invalid, k=n_invalid)):
        data.append({
            "id": n_valid + i,
            "sentence": sentence,
            "departure": "",
            "destination": "",
            "is_valid": False
        })

    return pd.DataFrame(data)

# Generer et sauvegarder
df = generate_dataset()
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Split train/test
train = df.sample(frac=0.8)
test = df.drop(train.index)

train.to_csv("data/dataset_train.csv", index=False)
test.to_csv("data/dataset_test.csv", index=False)
```

---

## Dataset Genere

Le script `scripts/make_dataset.py` genere automatiquement le dataset :

```bash
uv run python scripts/make_dataset.py
```

### Statistiques

| Ensemble | Phrases Valides | Phrases Invalides | Total |
|----------|-----------------|-------------------|-------|
| Train | ~5,600 | ~2,400 | 8,000 |
| Test | ~1,400 | ~600 | 2,000 |
| **Total** | **~7,000** | **~3,000** | **10,000** |

### Fichiers Generes

| Fichier | Format | Description |
|---------|--------|-------------|
| `data/train.spacy` | DocBin | Entrainement spaCy/CamemBERT |
| `data/test.spacy` | DocBin | Evaluation |
| `data/dataset_train.csv` | CSV | Analyse manuelle |
| `data/dataset_test.csv` | CSV | Evaluation metriques |

### Augmentation de Donnees

Le script applique plusieurs transformations pour augmenter la variete :

- **Minuscules** (20%) : `Je veux aller de Paris a Lyon` -> `je veux aller de paris a lyon`
- **Sans accents** (15%) : `de Béziers à Montpellier` -> `de beziers a montpellier`
- **Fautes de frappe** (10%) : `Marseille` -> `Marsielle`, `Marseeille`

### Templates Utilises

Les templates sont definis dans `data/templates/` :

- `valid.txt` : ~100 structures de phrases valides
- `invalid.txt` : ~70 phrases hors-sujet
- `ambiguous.txt` : ~50 phrases avec prenoms/villes ambigus

---

## Sources de Donnees SNCF

- **Liste des gares** : https://ressources.data.sncf.com/explore/dataset/liste-des-gares/
- **Referentiel des gares** : https://ressources.data.sncf.com/explore/dataset/referentiel-gares-voyageurs/
