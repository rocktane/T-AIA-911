# Architecture - Travel Order Resolver

## Pipeline Global

```
┌─────────────────────────────────────────────────────────────────────┐
│                           INPUT                                      │
│                  sentenceID,sentence (UTF-8)                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODULE NLP (coeur du projet)                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Preprocessing│ -> │ NER        │ -> │ Classification          │  │
│  │ - lowercase  │    │ - Detection│    │ Depart/Arrivee          │  │
│  │ - accents    │    │   des LOC  │    │                         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │    VALIDE     │       │   INVALIDE    │
            │ (Dep, Arr)    │       │               │
            └───────────────┘       └───────────────┘
                    │                       │
                    ▼                       │
┌─────────────────────────────────────────────────────────────────────┐
│                    MODULE GEOLOCALISATION                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Ville = Gare ?                                             │    │
│  │     OUI → Garder la ville                                   │    │
│  │     NON → Trouver la gare la plus proche (Haversine/KDTree) │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  Input:  (ville_dep, ville_arr)                                     │
│  Output: (gare_dep, dist_dep, gare_arr, dist_arr)                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODULE PATHFINDING                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ Graphe SNCF │ -> │ Dijkstra    │ -> │ Chemin optimal          │  │
│  │ (NetworkX)  │    │             │    │                         │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                     │
│  sentenceID,ville_dep,gare_dep,ville_arr,gare_arr,itineraire        │
│  ou  sentenceID,INVALID                                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Modules Detailles

### 1. Module NLP

**Objectif** : Extraire la ville de depart et la ville d'arrivee d'une phrase.

#### Approche Simple (Baseline)
```
Phrase -> Tokenization -> Detection villes (dictionnaire) -> Classification (regles)
```

#### Approche Intermediaire (spaCy)
```
Phrase -> spaCy NER -> Filtrer LOC -> Classification (regles + contexte)
```

#### Approche Avancee (CamemBERT)
```
Phrase -> Tokenization BERT -> Fine-tuned NER -> Labels DEP/ARR directement
```

**Entree** : `sentenceID,sentence`
**Sortie** : `sentenceID,Departure,Destination` ou `sentenceID,INVALID`

---

### 2. Module Pathfinding

**Objectif** : Trouver le chemin optimal entre deux gares.

```
                    ┌─────────┐
              ┌─────│  Paris  │─────┐
              │     └─────────┘     │
              │                     │
        ┌─────▼─────┐         ┌─────▼─────┐
        │   Lyon    │         │  Lille    │
        └─────┬─────┘         └───────────┘
              │
        ┌─────▼─────┐
        │ Marseille │
        └───────────┘
```

**Algorithme** : Dijkstra (NetworkX)
- Noeuds = Gares SNCF
- Aretes = Connexions ferroviaires
- Poids = Distance ou temps de trajet

**Entree** : `Departure, Destination`
**Sortie** : `[Departure, Step1, Step2, ..., Destination]`

---

## Structure du Code

```
src/
├── __init__.py
├── main.py              # Point d'entree, orchestration
├── nlp/
│   ├── __init__.py
│   ├── baseline.py      # Approche regex + dictionnaire
│   ├── spacy_ner.py     # Approche spaCy
│   └── camembert.py     # Approche Transformers (optionnel)
├── geo/
│   ├── __init__.py
│   ├── distance.py      # Formule de Haversine
│   ├── nearest_station.py # Recherche gare la plus proche
│   └── city_resolver.py # Resolution ville → gare
├── pathfinding/
│   ├── __init__.py
│   ├── graph.py         # Construction du graphe
│   └── search.py        # Algorithme Dijkstra
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py # Nettoyage du texte
│   └── metrics.py       # Calcul F1, Precision, Recall
└── data/
    ├── cities.py        # Dictionnaire des villes
    └── loader.py        # Chargement des donnees SNCF
```

---

## Formats de Donnees

### Input (stdin ou fichier)
```
1,Je veux aller de Paris a Lyon
2,Comment me rendre a Marseille depuis Bordeaux ?
3,Je veux aller de Cassis a Giverny
4,Il fait beau aujourd'hui
```

### Output NLP
```
1,Paris,Lyon
2,Bordeaux,Marseille
3,Cassis,Giverny
4,INVALID
```

### Output Geolocalisation
```
1,Paris,Paris,0,Lyon,Lyon,0
2,Bordeaux,Bordeaux,0,Marseille,Marseille,0
3,Cassis,Marseille,25,Giverny,Vernon,5
4,INVALID
```

### Output Final (avec Pathfinding)
```
1,Paris,Paris,Lyon,Lyon,"Paris→Lyon"
2,Bordeaux,Bordeaux,Marseille,Marseille,"Bordeaux→Toulouse→Marseille"
3,Cassis,Marseille,Giverny,Vernon,"Marseille→Lyon→Paris→Vernon"
4,INVALID
```

---

## Technologies Utilisees

| Composant | Technologie | Justification |
|-----------|-------------|---------------|
| NLP Baseline | Python + Regex | Simple, rapide a implementer |
| NLP Intermediaire | spaCy (fr_core_news_md) | NER pre-entraine pour le francais |
| NLP Avance | CamemBERT | BERT francais, fine-tunable |
| Geolocalisation | scipy.spatial.cKDTree | Recherche gare proche en O(log n) |
| Pathfinding | NetworkX | Librairie Python pour graphes |
| Donnees | Pandas | Manipulation CSV |

---

## Exemple de Flux Complet

**Input** : `1,Je voudrais aller de Cassis a Giverny`

```
Etape 1 - NLP:
  └─ Extraction: dep="Cassis", arr="Giverny"

Etape 2 - Geolocalisation:
  ├─ Cassis n'est pas une gare
  │   └─ Gare proche: Marseille (25 km)
  └─ Giverny n'est pas une gare
      └─ Gare proche: Vernon (5 km)

Etape 3 - Pathfinding:
  └─ Dijkstra(Marseille, Vernon) = [Marseille, Lyon, Paris, Vernon]

Output: 1,Cassis,Marseille,Giverny,Vernon,"Marseille→Lyon→Paris→Vernon"
```
