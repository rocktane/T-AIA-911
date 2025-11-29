# TO-DO - Travel Order Resolver

## Vue d'ensemble

Ce projet consiste a creer un systeme NLP capable de :
1. Determiner si une phrase est un ordre de voyage valide
2. Extraire la ville de depart et la ville d'arrivee
3. Trouver la gare la plus proche si la ville n'a pas de gare
4. Generer un itineraire ferroviaire avec les donnees SNCF

---

## Phase 1 : Preparation et Dataset

### 1.1 Recuperer les donnees SNCF (Gares)
- [ ] Telecharger la liste des gares SNCF (open data)
- [ ] Extraire : nom, ville, latitude, longitude
- [ ] Sauvegarder dans `data/gares_sncf.csv`

**Ressource** : https://ressources.data.sncf.com/

### 1.2 Recuperer la liste des villes francaises
- [ ] Telecharger la base des communes francaises (avec coordonnees GPS)
- [ ] Extraire : nom, code postal, latitude, longitude
- [ ] Sauvegarder dans `data/villes_france.csv`

**Ressources** :
- https://www.data.gouv.fr/fr/datasets/communes-de-france-base-des-codes-postaux/
- https://geo.api.gouv.fr/communes (API)

### 1.3 Creer le dataset d'entrainement
- [ ] Definir le format d'annotation : `<Dep>ville</Dep>` et `<Arr>ville</Arr>`
- [ ] Ecrire 50 phrases de base (templates)
- [ ] Generer ~2000 phrases par augmentation (variations de villes)
- [ ] **Inclure des villes SANS gare** dans le dataset
- [ ] Ajouter des phrases invalides (non-voyage)
- [ ] Ajouter des cas difficiles :
  - [ ] Prenoms = villes (Paris, Albert, Florence)
  - [ ] Villes avec mots communs (Port-Boulet)
  - [ ] Sans majuscules/accents
  - [ ] Fautes d'orthographe
  - [ ] Petites villes sans gare
- [ ] Separer en train/test (80/20)

**Objectif** : ~10 000 phrases, ~300 structures differentes

---

## Phase 2 : Solution Baseline (Simple)

### 2.1 Approche Regex + Dictionnaire
- [ ] Implementer la detection de villes par dictionnaire (gares + villes)
- [ ] Creer des regles regex pour identifier depart/arrivee :
  - `depuis X`, `de X`, `partant de X` → Depart
  - `vers X`, `a X`, `pour X` → Arrivee
- [ ] Gerer le format d'entree : `sentenceID,sentence`
- [ ] Gerer le format de sortie : `sentenceID,Departure,Destination`
- [ ] Tester et mesurer les metriques (F1, Precision, Recall)

**Fichier** : `src/baseline_nlp.py`

### 2.2 Amelioration avec spaCy
- [ ] Installer spaCy avec le modele francais (`fr_core_news_md`)
- [ ] Utiliser le NER de spaCy pour detecter les LOC (locations)
- [ ] Combiner avec les regles pour differencier depart/arrivee
- [ ] Comparer les metriques avec la baseline regex

**Fichier** : `src/spacy_nlp.py`

---

## Phase 3 : Gare la Plus Proche

### 3.1 Module de geolocalisation
- [ ] Charger les coordonnees des gares SNCF
- [ ] Charger les coordonnees des villes francaises
- [ ] Implementer la fonction de distance (Haversine)

```python
# Formule de Haversine pour distance entre 2 points GPS
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    # ... calcul de la distance
    return distance_km
```

### 3.2 Recherche de la gare la plus proche
- [ ] Pour une ville donnee, trouver la gare la plus proche
- [ ] Utiliser un KD-Tree pour optimiser la recherche (scipy.spatial)
- [ ] Retourner : (gare_proche, distance_km)

**Fichier** : `src/geo/nearest_station.py`

### 3.3 Integration dans le pipeline
- [ ] Si la ville extraite n'est pas une gare → chercher la gare la plus proche
- [ ] Ajouter l'info dans la sortie : `ville_demandee (gare: X, +Y km)`

**Exemple** :
```
Input:  "Je veux aller de Montargis a Saint-Tropez"
Output:
  - Depart: Montargis (gare: Montargis, 0 km)
  - Arrivee: Saint-Tropez (gare: Saint-Raphael, 35 km)
```

---

## Phase 4 : Pathfinding

### 4.1 Construction du graphe
- [ ] Importer les donnees SNCF dans NetworkX
- [ ] Creer les noeuds (gares) et aretes (connexions)
- [ ] Ajouter les poids (distance ou temps)

### 4.2 Algorithme de recherche
- [ ] Implementer Dijkstra avec NetworkX
- [ ] Format de sortie : `sentenceID,Dep,Step1,Step2,...,Arr`
- [ ] Gerer les cas sans chemin

**Fichier** : `src/pathfinding.py`

---

## Phase 5 : Integration

### 5.1 Pipeline complet
```
Phrase → NLP → (ville_dep, ville_arr)
                      ↓
              Ville = Gare ?
              /           \
           Oui            Non
            ↓              ↓
         Garder    Trouver gare proche
                           ↓
              (gare_dep, gare_arr)
                      ↓
               Pathfinding
                      ↓
                 Itineraire
```

- [ ] Creer le script principal qui enchaine tous les modules
- [ ] Lire depuis stdin ou fichier
- [ ] Gerer UTF-8
- [ ] Tester avec des fichiers d'exemple

**Fichier** : `src/main.py`

### 5.2 Format de sortie enrichi
```
sentenceID,ville_depart,gare_depart,ville_arrivee,gare_arrivee,itineraire
1,Montargis,Montargis,Saint-Tropez,Saint-Raphael,"Montargis→Paris→Marseille→Saint-Raphael"
```

### 5.3 Tests
- [ ] Creer un jeu de test avec phrases + resultats attendus
- [ ] Tester avec des villes sans gare
- [ ] Script d'evaluation automatique
- [ ] Documenter les resultats

---

## Phase 6 : Amelioration (si temps disponible)

### 6.1 Modele CamemBERT (Optionnel)
- [ ] Fine-tuner CamemBERT pour NER sur notre dataset
- [ ] Comparer avec la baseline
- [ ] Documenter l'evolution des metriques

### 6.2 Bonus
- [ ] Speech-to-Text avec Vosk
- [ ] Gestion des arrets intermediaires
- [ ] Interface CLI amelioree
- [ ] Proposer plusieurs gares proches (top 3)

---

## Phase 7 : Documentation

- [ ] `ARCHITECTURE.md` - Schema du pipeline
- [ ] `DATASET.md` - Description du dataset et processus de creation
- [ ] `NLP.md` - Choix techniques et resultats
- [ ] `PATHFINDING.md` - Algorithme utilise
- [ ] `GEO.md` - Recherche de gare la plus proche
- [ ] `METRICS.md` - Resultats des evaluations
- [ ] README.md final

---

## Metriques a suivre

| Metrique | Baseline | spaCy | CamemBERT |
|----------|----------|-------|-----------|
| Precision (Dep) | - | - | - |
| Recall (Dep) | - | - | - |
| F1 (Dep) | - | - | - |
| Precision (Arr) | - | - | - |
| Recall (Arr) | - | - | - |
| F1 (Arr) | - | - | - |
| Exact Match | - | - | - |

---

## Structure des fichiers

```
T-AIA-911/
├── data/
│   ├── gares_sncf.csv          # Gares avec coordonnees GPS
│   ├── villes_france.csv       # Toutes les villes francaises
│   ├── connexions_sncf.csv     # Connexions entre gares
│   ├── dataset_train.csv
│   └── dataset_test.csv
├── src/
│   ├── __init__.py
│   ├── main.py                 # Point d'entree
│   ├── nlp/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   └── spacy_ner.py
│   ├── geo/
│   │   ├── __init__.py
│   │   └── nearest_station.py  # Recherche gare proche
│   └── pathfinding/
│       ├── __init__.py
│       ├── graph.py
│       └── search.py
├── scripts/
│   ├── download_data.py        # Telecharger donnees SNCF/villes
│   ├── generate_dataset.py     # Generer le dataset
│   └── evaluate.py             # Evaluer les metriques
├── models/
│   └── (modeles entraines)
├── docs/
│   └── (documentation PDF finale)
├── TO-DO.md
├── ARCHITECTURE.md
├── DATASET.md
├── NLP.md
├── PATHFINDING.md
├── GEO.md
└── README.md
```

---

## Exemple de flux complet

**Input** : `1,Je voudrais aller de Cassis a Giverny`

**Etape 1 - NLP** :
- Ville depart detectee : Cassis
- Ville arrivee detectee : Giverny

**Etape 2 - Geolocalisation** :
- Cassis → Pas de gare → Gare proche : Marseille (25 km)
- Giverny → Pas de gare → Gare proche : Vernon (5 km)

**Etape 3 - Pathfinding** :
- Marseille → Lyon → Paris → Vernon

**Output** :
```
1,Cassis,Marseille,Giverny,Vernon,"Marseille→Lyon→Paris→Vernon"
```
