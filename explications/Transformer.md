# L'Architecture Transformer — Le Cerveau de CamemBERT

Prenons la phrase : "Je veux aller de Marseille à Paris"

Ce document explique comment CamemBERT, basé sur l'architecture Transformer, analyse cette phrase pour identifier les villes de départ et d'arrivée.

---

## Vue d'Ensemble : Encoder-Only vs Encoder-Decoder

Il existe deux grandes familles de Transformers :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARCHITECTURES TRANSFORMER                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ENCODER-DECODER (GPT, T5)          │  ENCODER-ONLY (BERT, CamemBERT)   │
│  ─────────────────────────          │  ──────────────────────────────   │
│  → Génère du texte                  │  → Comprend le texte              │
│  → Traduction, résumé               │  → Classification, NER            │
│  → Voit uniquement le passé         │  → Voit TOUT le contexte          │
│                                     │                                   │
│  "Je veux aller de [?]"             │  "Je veux aller de [Paris] à..."  │
│       ↓                             │       ↓                           │
│  Génère le mot suivant              │  Analyse chaque mot avec          │
│                                     │  son contexte complet             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**CamemBERT utilise l'architecture Encoder-Only** : il ne génère pas de texte, il le comprend. C'est parfait pour notre tâche de NER (Named Entity Recognition) où on doit classifier chaque mot.

---

## Le Mécanisme d'Attention — L'Innovation Clé

### Le Problème des RNN/LSTM

Avant les Transformers, on utilisait des RNN (Recurrent Neural Networks) et LSTM :

```
RNN/LSTM : Traitement SÉQUENTIEL
═══════════════════════════════

"Je" → "veux" → "aller" → "de" → "Marseille" → "à" → "Paris"
  ↓       ↓        ↓        ↓         ↓          ↓       ↓
 [H1] → [H2]  → [H3]  → [H4]  →    [H5]    → [H6] → [H7]

Problèmes :
┌────────────────────────────────────────────────────────────────┐
│ 1. SÉQUENTIEL : Chaque mot attend le précédent                 │
│    → Impossible de paralléliser sur GPU                        │
│                                                                │
│ 2. OUBLI : L'info de "Je" s'affaiblit en arrivant à "Paris"   │
│    → Le contexte lointain se perd                              │
│                                                                │
│ 3. UNIDIRECTIONNEL : "Marseille" ne voit pas "Paris"          │
│    → On ne sait pas encore que c'est une paire départ/arrivée  │
└────────────────────────────────────────────────────────────────┘
```

### La Solution : Self-Attention

Le Transformer résout tout cela avec le mécanisme de **Self-Attention** :

```
SELF-ATTENTION : Traitement PARALLÈLE et BIDIRECTIONNEL
════════════════════════════════════════════════════════

Chaque mot peut "regarder" TOUS les autres mots simultanément :

         "Je"  "veux" "aller" "de" "Marseille" "à" "Paris"
           │      │      │     │       │        │     │
           └──────┴──────┴─────┴───────┴────────┴─────┘
                              ↓
                    ┌─────────────────┐
                    │  SELF-ATTENTION │
                    │  (en parallèle) │
                    └─────────────────┘
                              ↓
           ┌──────┬──────┬─────┬───────┬────────┬─────┐
           │      │      │     │       │        │     │
         "Je"  "veux" "aller" "de" "Marseille" "à" "Paris"

         Chaque mot a maintenant une représentation
         enrichie par TOUT le contexte de la phrase
```

---

## Comment Fonctionne l'Attention ?

### Les Trois Vecteurs : Query, Key, Value

Pour chaque mot, on calcule trois vecteurs :

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MOT: "Marseille"                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query (Q) = "Que cherche-je ?"                                        │
│              → Marseille cherche des indices sur son rôle              │
│                                                                         │
│  Key (K)   = "Qui suis-je ?"                                           │
│              → Marseille s'identifie comme ville après "de"            │
│                                                                         │
│  Value (V) = "Quelle info j'apporte ?"                                 │
│              → Contenu sémantique de Marseille                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Le Calcul de l'Attention

Prenons "Marseille" qui veut comprendre son contexte :

```
Étape 1 : Calculer les scores d'attention
──────────────────────────────────────────

Score = Query(Marseille) · Key(autre_mot)

┌───────────┬─────────────────────────────────────────────┐
│   Mot     │  Score avec "Marseille"                     │
├───────────┼─────────────────────────────────────────────┤
│ "Je"      │  0.02  (peu pertinent)                      │
│ "veux"    │  0.05  (peu pertinent)                      │
│ "aller"   │  0.15  (verbe de déplacement → pertinent)   │
│ "de"      │  0.45  ← TRÈS IMPORTANT (indique le départ) │
│ "à"       │  0.25  (préposition d'arrivée)              │
│ "Paris"   │  0.08  (autre ville)                        │
└───────────┴─────────────────────────────────────────────┘


Étape 2 : Normaliser avec Softmax
─────────────────────────────────

Les scores deviennent des probabilités (somme = 1) :

         "de" ████████████████████████ 45%
       "aller" ███████ 15%
          "à" ██████████ 25%
       "Paris" ███ 8%
        "veux" ██ 5%
          "Je" █ 2%


Étape 3 : Pondérer les Values
─────────────────────────────

Nouvelle représentation de "Marseille" =
    0.45 × Value(de) + 0.25 × Value(à) + 0.15 × Value(aller) + ...

→ "Marseille" intègre fortement l'info de "de"
→ Le modèle comprend que c'est un DÉPART
```

---

## Les 12 Couches de CamemBERT

CamemBERT empile **12 couches** identiques. Chaque couche raffine la compréhension :

```
ENTRÉE: "Je veux aller de Marseille à Paris"
        ↓
┌───────────────────────────────────────────────────────────────┐
│  COUCHE 1 — Reconnaissance basique                            │
│  ─────────────────────────────────                            │
│  → "Marseille" et "Paris" = probablement des noms propres     │
│  → "de" et "à" = prépositions                                 │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│  COUCHE 2-3 — Patterns syntaxiques                            │
│  ─────────────────────────────────────                        │
│  → "aller de X à Y" = structure de déplacement                │
│  → X après "de" = origine                                     │
│  → Y après "à" = destination                                  │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│  COUCHE 4-6 — Relations sémantiques                           │
│  ─────────────────────────────────────                        │
│  → "Marseille" = entité géographique (ville)                  │
│  → "veux aller" = intention de voyage                         │
│  → Lien fort entre "Marseille" et "Paris" (paire voyage)      │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│  COUCHE 7-9 — Compréhension contextuelle                      │
│  ───────────────────────────────────────                      │
│  → "de Marseille" = point de DÉPART confirmé                  │
│  → "à Paris" = point d'ARRIVÉE confirmé                       │
│  → Distinction claire des rôles                               │
└───────────────────────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────────────────────┐
│  COUCHE 10-12 — Raffinement pour la tâche                     │
│  ────────────────────────────────────────                     │
│  → Représentations optimisées pour le NER                     │
│  → "Marseille" → prêt pour label B-DEPART                     │
│  → "Paris" → prêt pour label B-ARRIVEE                        │
└───────────────────────────────────────────────────────────────┘
        ↓
SORTIE: Représentations contextualisées (768 dimensions par mot)
```

### Multi-Head Attention : Plusieurs Points de Vue

Chaque couche utilise **12 têtes d'attention** en parallèle :

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION (12 têtes)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Tête 1  → Focus sur les relations syntaxiques (sujet-verbe)           │
│  Tête 2  → Focus sur les prépositions ("de", "à")                      │
│  Tête 3  → Focus sur les noms propres (majuscules)                     │
│  Tête 4  → Focus sur la distance entre mots                            │
│  Tête 5  → Focus sur les patterns "de X à Y"                           │
│  ...                                                                    │
│  Tête 12 → Focus sur les relations longue distance                     │
│                                                                         │
│  → Chaque tête capture un aspect différent du langage                  │
│  → Les résultats sont combinés pour une vue complète                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bidirectionnalité : Le Contexte Complet

### Le Problème de l'Unidirectionnalité

Comparons avec un modèle unidirectionnel (comme GPT) :

```
UNIDIRECTIONNEL (gauche → droite) :
───────────────────────────────────

"Je veux aller de Marseille à Paris"

Quand on analyse "Marseille" :
→ On voit : "Je veux aller de"
→ On ne voit PAS : "à Paris"
→ Problème : Comment savoir que c'est un DÉPART sans voir l'ARRIVÉE ?


BIDIRECTIONNEL (CamemBERT) :
────────────────────────────

"Je veux aller de Marseille à Paris"

Quand on analyse "Marseille" :
→ On voit GAUCHE : "Je veux aller de"
→ On voit DROITE : "à Paris"
→ On comprend la structure complète "de X à Y"
```

### Exemple Concret : Ordre Inversé

```
Phrase A: "Je veux aller de Marseille à Paris"
Phrase B: "À Paris, je veux aller depuis Marseille"

MODÈLE BIDIRECTIONNEL :
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Phrase A: "Marseille" voit "de" à gauche ET "à Paris" à droite │
│            → Conclusion : DÉPART                                 │
│                                                                  │
│  Phrase B: "Marseille" voit "depuis" à gauche ET "À Paris" loin │
│            à gauche                                              │
│            → Conclusion : toujours DÉPART (grâce à "depuis")    │
│                                                                  │
│  Dans les deux cas, le modèle comprend correctement !           │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Parallélisme vs Séquentiel

### Comparaison des Performances

```
TRAITEMENT D'UNE PHRASE DE 7 MOTS
═════════════════════════════════

RNN/LSTM (Séquentiel) :
───────────────────────
Temps = 7 étapes × temps_par_mot

   t=1    t=2    t=3    t=4    t=5    t=6    t=7
   "Je" → "veux" → "aller" → "de" → "Mars." → "à" → "Paris"

   → Chaque mot attend le précédent
   → Impossible d'utiliser les milliers de coeurs GPU


TRANSFORMER (Parallèle) :
─────────────────────────
Temps = 1 étape (tous les mots en parallèle)

   t=1
   "Je"  "veux"  "aller"  "de"  "Marseille"  "à"  "Paris"
    ↓      ↓       ↓       ↓        ↓         ↓      ↓
   [H1]  [H2]    [H3]    [H4]     [H5]      [H6]   [H7]

   → Tous les calculs en même temps
   → Exploitation maximale du GPU


GAIN DE VITESSE :
─────────────────
┌────────────────────────────────────────────────────────────────┐
│  Phrase de 100 mots :                                          │
│  - RNN  : ~100 étapes séquentielles                            │
│  - Transformer : ~1 étape parallèle (avec 12 couches)          │
│                                                                │
│  Sur GPU moderne : Transformer 10-100x plus rapide             │
└────────────────────────────────────────────────────────────────┘
```

### L'Astuce des Positional Embeddings

Sans traitement séquentiel, comment le modèle sait-il l'ordre des mots ?

```
POSITIONAL EMBEDDINGS
═════════════════════

Chaque position reçoit un vecteur unique :

Position 0 : [sin(0), cos(0), sin(0/10), cos(0/10), ...]
Position 1 : [sin(1), cos(1), sin(1/10), cos(1/10), ...]
Position 2 : [sin(2), cos(2), sin(2/10), cos(2/10), ...]
...

Ces vecteurs sont AJOUTÉS aux embeddings des mots :

┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  Embedding("Marseille") + Positional(4) = Embedding_final     │
│                                                                │
│  → Le modèle sait que "Marseille" est en position 4           │
│  → Il peut apprendre que position après "de" = DÉPART         │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Application au NER dans notre Projet

### Le Pipeline Complet

```
ENTRÉE : "Je veux aller de Marseille à Paris"
         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  TOKENIZATION (CamemBERT Tokenizer)                                     │
│  ───────────────────────────────────                                    │
│  → ["▁Je", "▁veux", "▁aller", "▁de", "▁Marseille", "▁à", "▁Paris"]     │
│  → Conversion en IDs : [147, 1823, 5621, 13, 8234, 22, 1456]           │
└─────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  12 COUCHES TRANSFORMER                                                 │
│  ──────────────────────                                                 │
│  → Self-Attention bidirectionnelle                                      │
│  → Multi-Head Attention (12 têtes)                                      │
│  → Feed-Forward Networks                                                │
│  → Layer Normalization                                                  │
└─────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  REPRÉSENTATIONS CONTEXTUALISÉES                                        │
│  ───────────────────────────────                                        │
│  → Chaque token : vecteur de 768 dimensions                            │
│  → "Marseille" : enrichi par contexte "de ... à Paris"                 │
│  → "Paris" : enrichi par contexte "de Marseille à ..."                 │
└─────────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│  CLASSIFICATION NER (couche linéaire)                                   │
│  ────────────────────────────────────                                   │
│  → 768 dimensions → 7 classes                                           │
│  → Classes : O, B-DEPART, I-DEPART, B-ARRIVEE, I-ARRIVEE, B-VIA, I-VIA │
└─────────────────────────────────────────────────────────────────────────┘
         ↓
SORTIE :
┌───────────────┬────────────┐
│    Token      │   Label    │
├───────────────┼────────────┤
│ "Je"          │ O          │
│ "veux"        │ O          │
│ "aller"       │ O          │
│ "de"          │ O          │
│ "Marseille"   │ B-DEPART   │  ← Détecté comme DÉPART
│ "à"           │ O          │
│ "Paris"       │ B-ARRIVEE  │  ← Détecté comme ARRIVÉE
└───────────────┴────────────┘
```

### Référence au Code

Dans `src/nlp/camembert_ner.py` :

```python
# Ligne 275-280 : Chargement du modèle Transformer pré-entraîné
self.model = AutoModelForTokenClassification.from_pretrained(
    self.config.model_name,  # "camembert-base"
    num_labels=NUM_LABELS,   # 7 classes pour le NER
    id2label=ID2LABEL,       # {0: "O", 1: "B-DEPART", ...}
    label2id=LABEL2ID,       # {"O": 0, "B-DEPART": 1, ...}
)
```

Le modèle `camembert-base` contient :
- **12 couches Transformer** (identiques à BERT)
- **12 têtes d'attention** par couche
- **768 dimensions** par représentation
- **110M paramètres** au total

---

## Récapitulatif : Transformer vs RNN/LSTM

```
┌─────────────────────────────────────────────────────────────────────────┐
│              COMPARAISON TRANSFORMER vs RNN/LSTM                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Critère              │  RNN/LSTM          │  Transformer              │
│  ─────────────────────┼────────────────────┼───────────────────────────│
│  Direction            │  Unidirectionnel   │  Bidirectionnel           │
│                       │  (ou bi-LSTM lent) │  (natif)                  │
│                                                                         │
│  Traitement           │  Séquentiel        │  Parallèle                │
│                       │  (1 mot à la fois) │  (tous en même temps)     │
│                                                                         │
│  Contexte lointain    │  Se dégrade        │  Accès direct             │
│                       │  (vanishing grad)  │  (self-attention)         │
│                                                                         │
│  Vitesse GPU          │  Lent              │  Très rapide              │
│                       │  (pas parallèle)   │  (massivement parallèle)  │
│                                                                         │
│  Capture patterns     │  Difficile         │  Multi-head attention     │
│  multiples            │                    │  (12 vues différentes)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ★ Insight ─────────────────────────────────────

**Pourquoi 768 dimensions ?** C'est la taille standard de BERT-base. Chaque dimension capture un aspect du sens du mot. Par exemple, certaines dimensions pourraient encoder "est-ce un lieu ?", "est-ce après une préposition de mouvement ?", etc.

**Le pré-entraînement MLM** : CamemBERT a été pré-entraîné sur 138 Go de texte français avec la tâche "Masked Language Model" (prédire les mots masqués). Il a appris les patterns du français AVANT notre fine-tuning NER. C'est pourquoi il comprend déjà que "de X à Y" indique un déplacement.

**Complexité O(n²)** : L'attention calcule tous les paires de mots, donc O(n²) en mémoire. Pour une phrase de 100 mots = 10,000 calculs d'attention. C'est pourquoi on limite souvent à 128-512 tokens maximum.

─────────────────────────────────────────────────
