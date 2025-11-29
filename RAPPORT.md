# RAPPORT D'ÉVALUATION - Travel Order Resolver
## Projet NLP - MSc Pro Spécialité IA - Epitech

---

## 1. Contexte et Objectifs du Projet

### 1.1 Description du Sujet

Le projet **Travel Order Resolver** est un système de traitement du langage naturel (NLP) dont l'objectif est d'extraire des villes de départ et d'arrivée depuis des phrases en français, afin de générer des itinéraires ferroviaires SNCF.

**Pipeline attendu :**
```
Phrase utilisateur → NLP (extraction) → Géolocalisation → Pathfinding → Itinéraire
```

### 1.2 Compétences Évaluées (MSc Pro IA)

Pour un projet de Master 2 spécialité Intelligence Artificielle, les attentes principales sont :
- **Méthodologie ML/IA** : création de dataset, entraînement, évaluation, expérimentation
- **Compréhension des modèles** : NER, Transformers, fine-tuning
- **Métriques quantitatives** : Precision, Recall, F1-score
- **Comparaison de modèles** : baseline vs modèles avancés

### 1.3 Documents de Référence

- `project.pdf` : Sujet principal
- `bootstrap.pdf` : Introduction au NER
- `Fiche ressources.pdf` : Ressources SNCF et NLP

---

## 2. Barème de Notation (100 points)

### 2.1 Philosophie du Barème

Ce barème donne la **priorité aux compétences IA** (45% des points) car il s'agit d'un projet de spécialité IA. L'ingénierie logicielle, bien qu'importante, est secondaire par rapport à la démonstration des compétences en Machine Learning.

### 2.2 Répartition des Points

| Section | Points | Pourcentage |
|---------|--------|-------------|
| **1. Méthodologie et Compétences IA** | **45** | **45%** |
| 2. Implémentation NLP | 20 | 20% |
| 3. Modules Complémentaires (Geo/Pathfinding) | 15 | 15% |
| 4. Qualité Logicielle | 15 | 15% |
| 5. Livrables et Format I/O | 5 | 5% |
| **TOTAL** | **100** | **100%** |

### 2.3 Détail du Barème

#### Section 1 : Méthodologie et Compétences IA (45 points)

| Critère | Points | Description |
|---------|--------|-------------|
| Dataset annotée | 10 | Création d'un dataset de qualité avec annotations NER |
| Process d'entraînement | 12 | Fine-tuning documenté, hyperparamètres, courbes d'apprentissage |
| Métriques d'évaluation | 10 | Calcul et documentation de Precision/Recall/F1 par entité |
| Expérimentation | 8 | Comparaison baseline vs modèles avancés, ablation study |
| CamemBERT/Transformers | 5 | Implémentation d'un modèle Transformer fine-tuné |

#### Section 2 : Implémentation NLP (20 points)

| Critère | Points | Description |
|---------|--------|-------------|
| Preprocessing | 5 | Normalisation, gestion des accents, abréviations |
| Baseline fonctionnel | 5 | Solution regex/dictionnaire fonctionnelle |
| Modèle avancé | 5 | spaCy ou équivalent avec capacité de training |
| Cas difficiles | 5 | Prénoms-villes, noms composés, ordre inverse, typos |

#### Section 3 : Modules Complémentaires (15 points)

| Critère | Points | Description |
|---------|--------|-------------|
| Géolocalisation | 8 | Haversine, KD-Tree, résolution ville→gare |
| Pathfinding | 7 | Dijkstra, graphe SNCF, calcul d'itinéraire |

#### Section 4 : Qualité Logicielle (15 points)

| Critère | Points | Description |
|---------|--------|-------------|
| Tests | 5 | Couverture de tests, tests unitaires et d'intégration |
| Documentation | 5 | README, docstrings, architecture documentée |
| Architecture | 5 | Modularité, type hints, code propre |

#### Section 5 : Livrables et Format I/O (5 points)

| Critère | Points | Description |
|---------|--------|-------------|
| Spécifications | 5 | Format CSV conforme, encodage UTF-8, interfaces CLI/Web |

### 2.4 Pénalités

| Pénalité | Points | Condition |
|----------|--------|-----------|
| CamemBERT absent | **-10** | Mentionné dans le scope mais non implémenté |
| Métriques non calculées | **-5** | Tableau de métriques vide ou absent |
| Dataset non généré | **-5** | Scripts prêts mais datasets non créés |

---

## 3. Analyse Détaillée par Critère

### 3.1 Méthodologie et Compétences IA

#### 3.1.1 Dataset Annotée (10 points max)

**Constat :**
- Script `scripts/make_dataset.py` présent (412 lignes)
- Génération de 7000 phrases valides + 3000 invalides prévue
- Templates variés (`departure_first.txt`, `arrival_first.txt`)
- Application de bruit (lowercase, accents, typos)
- Format spaCy DocBin prévu

**Problème majeur :**
```bash
$ ls data/*.spacy
ls: data/*.spacy: No such file or directory
```
Les fichiers `train.spacy` et `test.spacy` **n'existent pas**. Le script n'a jamais été exécuté.

**Note : 4/10** - Infrastructure prête mais non exécutée.

#### 3.1.2 Process d'Entraînement (12 points max)

**Constat :**
- Aucun code de fine-tuning présent
- Utilisation exclusive de modèles pré-entraînés (`fr_core_news_md`)
- Pas de documentation du process d'entraînement
- Pas de configuration d'hyperparamètres
- Pas de courbes d'apprentissage

**Preuve dans le code :**
```python
# src/nlp/spacy_ner.py ligne 57
self.nlp = spacy.load("fr_core_news_md")  # Modèle pré-entraîné, pas de custom training
```

**Note : 1/12** - Le code charge un modèle existant sans aucun entraînement.

#### 3.1.3 Métriques d'Évaluation (10 points max)

**Constat :**
Le fichier `CLAUDE.md` contient un tableau de métriques **vide** :

```markdown
| Metrique | Baseline | spaCy | CamemBERT |
|----------|----------|-------|-----------|
| Precision (Dep) | - | - | - |
| Recall (Dep) | - | - | - |
| F1 (Dep) | - | - | - |
...
```

- Pas de script `evaluate.py`
- Aucune métrique calculée
- Pas de confusion matrix
- Pas de rapport d'évaluation

**Note : 0/10** - Aucune évaluation quantitative réalisée.

#### 3.1.4 Expérimentation (8 points max)

**Constat :**
- Aucune comparaison documentée baseline vs spaCy
- Pas de test A/B sur différentes configurations
- Pas d'ablation study
- Pas de validation croisée

**Note : 1/8** - Les deux modèles existent mais jamais comparés formellement.

#### 3.1.5 CamemBERT/Transformers (5 points max)

**Constat :**
Le fichier `src/nlp/camembert_ner.py` **n'existe pas** :
```bash
$ ls src/nlp/
__init__.py  baseline.py  preprocessing.py  spacy_ner.py
# camembert_ner.py ABSENT
```

Pourtant, les dépendances sont présentes dans `pyproject.toml` :
```toml
transformers = ">=4.36.0"
torch = ">=2.1.0"
```

Et le fichier `CLAUDE.md` mentionne CamemBERT comme prévu dans le scope.

**Note : 0/5** - Engagement non tenu.

---

### 3.2 Implémentation NLP

#### 3.2.1 Preprocessing (5 points max)

**Constat :**
Le fichier `src/nlp/preprocessing.py` (123 lignes) implémente :
- `remove_accents()` : Suppression des accents
- `normalize_whitespace()` : Normalisation des espaces
- `expand_abbreviations()` : "st" → "saint", "ste" → "sainte"
- `preprocess_text()` : Pipeline configurable
- `normalize_city_name()` : Normalisation pour matching

**Qualité : Excellente**

**Note : 5/5**

#### 3.2.2 Baseline Fonctionnel (5 points max)

**Constat :**
Le fichier `src/nlp/baseline.py` (159 lignes) implémente :
- 6 patterns regex pour le départ
- 3 patterns regex pour l'arrivée
- Validation contre dictionnaire de villes
- Fallback positionnel

**Tests passants :** 7/7

**Note : 5/5**

#### 3.2.3 Modèle Avancé (5 points max)

**Constat :**
Le fichier `src/nlp/spacy_ner.py` (396 lignes) implémente :
- NER avec spaCy pré-entraîné
- Classification rule-based du contexte
- Fallback sur dictionnaire
- Support pour custom model (non utilisé)

**Limitation :** Pas de custom training, utilisation du modèle générique.

**Note : 4/5**

#### 3.2.4 Cas Difficiles (5 points max)

| Cas | Implémenté | Fonctionnel |
|-----|------------|-------------|
| Prénoms = villes (Albert, Nancy) | Oui | Partiellement |
| Noms composés (Saint-Pierre) | Oui | Oui |
| Sans majuscules | Oui | Oui |
| Sans accents | Oui | Oui |
| **Fautes d'orthographe** | **Non** | - |
| Ordre inverse | Oui | Oui |

**Note : 3/5** - Les typos ne sont pas gérées.

---

### 3.3 Modules Complémentaires

#### 3.3.1 Géolocalisation (8 points max)

**Implémentation :**
- `distance.py` : Haversine (correct, testé)
- `nearest_station.py` : KD-Tree avec scipy (O(log n))
- `city_resolver.py` : Cascade logique station → ville → fuzzy → nearest

**Qualité : Excellente**

**Note : 8/8**

#### 3.3.2 Pathfinding (7 points max)

**Implémentation :**
- `graph.py` : RailwayGraph avec NetworkX
- `dijkstra.py` : Dijkstra via `nx.dijkstra_path()`

**Bug identifié :**
```python
# src/pathfinding/dijkstra.py ligne 91
distance = edge_data.get("distance", 0) or 0  # Retourne toujours 0 si pas d'attribut "distance"
```
Les tests échouent car `total_distance = 0.0`.

**Note : 6/7** - Bug mineur mais fonctionnel.

---

### 3.4 Qualité Logicielle

#### 3.4.1 Tests (5 points max)

**Résultats :**
```
Total: 24 tests
Passés: 22 (91.7%)
Échoués: 2 (bug pathfinding distance)
```

**Limitation :** Pas de tests pour SpacyNER.

**Note : 4/5**

#### 3.4.2 Documentation (5 points max)

**Fichiers présents :**
- `README.md` : Installation et usage
- `CLAUDE.md` : Spécifications complètes
- `ARCHITECTURE.md` : Diagrammes du pipeline
- `NLP.md`, `GEO.md`, `PATHFINDING.md`, `DATASET.md` : Détails techniques

**Qualité : Excellente** (7 fichiers .md détaillés)

**Note : 5/5**

#### 3.4.3 Architecture (5 points max)

**Points positifs :**
- Modularité : séparation NLP/Geo/Pathfinding
- Type hints complets
- Dataclasses pour les résultats
- Code propre et lisible

**Note : 5/5**

---

### 3.5 Livrables et Format I/O

**Format d'entrée (conforme) :**
```csv
sentenceID,sentence
1,Je veux aller de Paris à Lyon
```

**Format de sortie (conforme) :**
```csv
sentenceID,ville_dep,gare_dep,ville_arr,gare_arr,itineraire
1,Paris,Paris Gare de Lyon,Lyon,Lyon Part-Dieu,"Paris→Lyon"
```

**Interfaces :**
- CLI : `uv run python -m src.main`
- Web : FastAPI avec interface HTML

**Note : 5/5**

---

## 4. Évaluation et Notation

### 4.1 Tableau Récapitulatif

#### Section 1 : Méthodologie IA (45 pts max)

| Critère | Max | Note | Justification |
|---------|-----|------|---------------|
| Dataset annotée | 10 | 4 | Script prêt mais datasets non générés |
| Entraînement documenté | 12 | 1 | Aucun fine-tuning, modèles pré-entraînés uniquement |
| Métriques Precision/Recall/F1 | 10 | 0 | Tableau vide, evaluate.py inexistant |
| Expérimentation | 8 | 1 | Aucune comparaison documentée |
| CamemBERT/Transformers | 5 | 0 | Non implémenté |
| **Sous-total** | **45** | **6** | **13%** |

#### Section 2 : Implémentation NLP (20 pts max)

| Critère | Max | Note | Justification |
|---------|-----|------|---------------|
| Preprocessing | 5 | 5 | Excellent |
| Baseline | 5 | 5 | Fonctionnel |
| Modèle avancé | 5 | 4 | spaCy OK, pas de custom training |
| Cas difficiles | 5 | 3 | Typos non gérées |
| **Sous-total** | **20** | **17** | **85%** |

#### Section 3 : Modules Complémentaires (15 pts max)

| Critère | Max | Note | Justification |
|---------|-----|------|---------------|
| Géolocalisation | 8 | 8 | Excellent (KD-Tree, Haversine) |
| Pathfinding | 7 | 6 | Bon, bug mineur distance |
| **Sous-total** | **15** | **14** | **93%** |

#### Section 4 : Qualité Logicielle (15 pts max)

| Critère | Max | Note | Justification |
|---------|-----|------|---------------|
| Tests | 5 | 4 | 91.7%, pas de tests SpacyNER |
| Documentation | 5 | 5 | Excellente |
| Architecture | 5 | 5 | Modulaire, propre |
| **Sous-total** | **15** | **14** | **93%** |

#### Section 5 : Livrables (5 pts max)

| Critère | Max | Note | Justification |
|---------|-----|------|---------------|
| Specs et interface | 5 | 5 | Conforme |
| **Sous-total** | **5** | **5** | **100%** |

### 4.2 Pénalités Appliquées

| Pénalité | Points | Justification |
|----------|--------|---------------|
| CamemBERT absent | -10 | Mentionné dans CLAUDE.md, dépendances ajoutées, jamais implémenté |
| Métriques non calculées | -5 | Tableau vide, aucune évaluation quantitative |
| Dataset non généré | -5 | train.spacy et test.spacy inexistants |
| **Total pénalités** | **-20** | |

### 4.3 Calcul de la Note Finale

| Section | Points |
|---------|--------|
| Méthodologie IA | 6/45 |
| Implémentation NLP | 17/20 |
| Modules complémentaires | 14/15 |
| Qualité logicielle | 14/15 |
| Livrables | 5/5 |
| **Sous-total** | **56/100** |
| Pénalités | **-20** |
| **NOTE FINALE** | **36/100** |

---

## 5. Points Forts du Projet

### 5.1 Excellence en Ingénierie Logicielle

1. **Architecture modulaire exemplaire** : Séparation claire des responsabilités (NLP, Geo, Pathfinding)
2. **Documentation complète** : 7 fichiers .md couvrant tous les aspects
3. **Code propre** : Type hints, dataclasses, conventions respectées
4. **Tests présents** : 91.7% de réussite, bonne couverture de base

### 5.2 Modules Complémentaires de Qualité

1. **Géolocalisation optimisée** : KD-Tree pour recherche O(log n)
2. **Haversine correct** : Formule géodésique bien implémentée
3. **Pathfinding fonctionnel** : Dijkstra via NetworkX

### 5.3 Pipeline Fonctionnel

1. **CLI complet** : Lecture stdin/fichier, options de configuration
2. **Interface web** : FastAPI avec highlights visuels
3. **Format I/O conforme** : Respect des spécifications du sujet

---

## 6. Lacunes Identifiées (Focus IA)

### 6.1 Absence Totale de Cycle ML

Le projet ne démontre **aucune compétence en Machine Learning** :

| Étape ML | Attendu | Réalité |
|----------|---------|---------|
| Data Collection | Dataset annotée | Script prêt, non exécuté |
| Data Preprocessing | Train/Val/Test split | Non réalisé |
| Model Training | Fine-tuning | Aucun |
| Hyperparameter Tuning | Grid search, etc. | Aucun |
| Evaluation | Precision/Recall/F1 | Non calculé |
| Model Comparison | Baseline vs Advanced | Non documenté |

### 6.2 CamemBERT : Promesse Non Tenue

Le fichier `CLAUDE.md` mentionne explicitement CamemBERT :
```markdown
- **NLP**: spaCy (baseline) + CamemBERT (fine-tuning)
```

Les dépendances sont ajoutées :
```toml
transformers = ">=4.36.0"
torch = ">=2.1.0"
```

Mais le fichier `camembert_ner.py` **n'existe pas**. C'est un engagement non tenu qui justifie une pénalité forte.

### 6.3 Métriques Vides

Le sujet insiste sur l'importance des métriques :
> "You obviously need metrics to measure it!"

Le tableau dans CLAUDE.md est **entièrement vide** :
```
| Precision (Dep) | - | - | - |
| Recall (Dep) | - | - | - |
```

Cela démontre une incompréhension des attentes d'un projet ML.

### 6.4 Utilisation de Modèles "Off-the-Shelf"

Le projet utilise exclusivement des modèles pré-entraînés sans adaptation :
- spaCy `fr_core_news_md` : modèle générique
- Pas de custom training sur le domaine ferroviaire
- Pas d'amélioration par rapport à l'état de l'art

Pour un MSc Pro spécialité IA, on attend une **adaptation au domaine** via fine-tuning.

---

## 7. Recommandations d'Amélioration

### 7.1 Priorité CRITIQUE : Cycle ML Complet

**Actions à réaliser :**

1. **Générer le dataset**
   ```bash
   uv run python scripts/make_dataset.py
   ```
   Vérifier la création de `data/train.spacy` et `data/test.spacy`

2. **Créer un script d'évaluation**
   Fichier : `scripts/evaluate.py`
   - Charger le modèle et le dataset de test
   - Calculer Precision, Recall, F1 par entité (DEPART, ARRIVEE)
   - Calculer Exact Match
   - Générer une confusion matrix

3. **Documenter les résultats**
   Remplir le tableau de métriques dans CLAUDE.md avec les vraies valeurs

### 7.2 Priorité IMPORTANTE : CamemBERT

**Actions à réaliser :**

1. **Créer `src/nlp/camembert_ner.py`**
   - Utiliser `transformers` avec `camembert-base`
   - Adapter pour token classification (NER)
   - Implémenter le fine-tuning sur le dataset custom

2. **Entraîner et évaluer**
   - Entraîner sur train.spacy
   - Évaluer sur test.spacy
   - Comparer avec baseline et spaCy

3. **Documenter**
   - Hyperparamètres utilisés
   - Courbes d'apprentissage (loss, metrics)
   - Résultats comparatifs

### 7.3 Priorité RECOMMANDÉE : Expérimentation

**Actions à réaliser :**

1. **Validation croisée**
   - Tester différents splits (80/20, 70/30, k-fold)
   - Documenter la variance des résultats

2. **Ablation study**
   - Impact du preprocessing
   - Impact de la taille du dataset
   - Impact des hyperparamètres

3. **Visualisations**
   - Confusion matrix par modèle
   - Courbes precision-recall
   - Exemples d'erreurs analysés

### 7.4 Priorité MINEURE : Corrections Techniques

1. **Bug pathfinding** : Corriger le calcul de `total_distance` dans `dijkstra.py`
2. **Tests SpacyNER** : Ajouter des tests unitaires
3. **Fuzzy matching NLP** : Implémenter la gestion des fautes d'orthographe

---

## 8. Conclusion et Note Finale

### 8.1 Synthèse

Le projet **Travel Order Resolver** présente une **dichotomie marquée** entre :

**Excellence technique :**
- Architecture logicielle exemplaire
- Documentation complète et professionnelle
- Modules géolocalisation et pathfinding de qualité
- Pipeline fonctionnel bout-en-bout

**Insuffisance en compétences IA :**
- Aucun cycle ML complet
- Pas de fine-tuning
- Métriques non calculées
- CamemBERT non implémenté malgré l'engagement

### 8.2 Appréciation Qualitative

Pour un projet de **MSc Pro spécialité IA**, les attentes portent principalement sur la démonstration des compétences en Machine Learning. Ce projet, bien qu'étant un excellent travail d'ingénierie logicielle, **ne démontre pas les compétences fondamentales attendues** d'un étudiant en Master 2 IA :

- Pas de création et utilisation de dataset d'entraînement
- Pas de processus de fine-tuning documenté
- Pas d'évaluation quantitative des performances
- Pas de comparaison expérimentale de modèles

Le projet ressemble davantage à un travail de **développement logiciel** qu'à un travail de **Data Science/ML**.

### 8.3 Note Finale

| | |
|---|---|
| **Note brute** | 56/100 |
| **Pénalités** | -20 |
| **NOTE FINALE** | **36/100** |
| **Mention** | **Insuffisant** |

### 8.4 Message aux Étudiants

Ce projet démontre de réelles compétences en développement logiciel. Cependant, pour un projet de spécialité IA, il est **impératif** de :

1. **Créer et utiliser un dataset** d'entraînement
2. **Entraîner au moins un modèle** (fine-tuning)
3. **Calculer et documenter les métriques** (Precision, Recall, F1)
4. **Comparer les approches** de manière quantitative
5. **Tenir ses engagements** (si CamemBERT est mentionné, l'implémenter)

Le travail d'implémentation des modules baseline, géolocalisation et pathfinding est de qualité. Avec un effort supplémentaire sur la partie ML, ce projet pourrait obtenir une note significativement meilleure.

---

*Rapport généré le 26 novembre 2025*
*Évaluateur : Enseignant MSc Pro IA - Epitech*
