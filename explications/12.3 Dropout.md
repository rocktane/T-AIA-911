⏺ Le Dropout avec des exemples concrets

Prenons l'entraînement de notre modèle NER sur la phrase : "Je veux aller de Marseille à Paris"

---
Le problème : le sur-apprentissage

Sans régularisation, le modèle peut "mémoriser" les exemples d'entraînement au lieu d'apprendre des patterns généraux.

┌─────────────────────────────────────────────────────────────────┐
│  EXEMPLE DE SUR-APPRENTISSAGE                                   │
│                                                                 │
│  Le modèle apprend :                                            │
│  "Si le mot est 'Marseille' ET précédé de 'de' ET suivi de 'à' │
│   ET la phrase fait 7 mots ET commence par 'Je'..."            │
│                                                                 │
│  Au lieu de :                                                   │
│  "Si le mot est un nom propre précédé de 'de' → DÉPART"        │
│                                                                 │
│  → Le modèle échoue sur les nouvelles phrases                   │
└─────────────────────────────────────────────────────────────────┘

---
La solution : le Dropout

Pendant l'entraînement, on "éteint" aléatoirement certains neurones :

┌─────────────────────────────────────────────────────────────────┐
│  dropout = 0.1 (10%)                                            │
│                                                                 │
│  AVANT DROPOUT (tous les neurones actifs)                       │
│                                                                 │
│  Entrée:  [o]───[o]───[o]───[o]───[o]  Couche cachée            │
│              \  / \  / \  / \  /                                │
│               \/   \/   \/   \/                                 │
│  Sortie:     [o]  [o]  [o]  [o]                                 │
│                                                                 │
│  APRÈS DROPOUT (10% des neurones éteints)                       │
│                                                                 │
│  Entrée:  [o]───[X]───[o]───[o]───[o]  ← [X] = neurone éteint   │
│              \     \  / \  /                                    │
│               \     \/   \/                                     │
│  Sortie:     [o]  [o]  [o]  [o]                                 │
│                                                                 │
│  Le neurone éteint ne contribue pas à la prédiction             │
└─────────────────────────────────────────────────────────────────┘

---
Pourquoi ça marche ?

Le dropout force le réseau à être REDONDANT :

┌─────────────────────────────────────────────────────────────────┐
│  SANS DROPOUT                                                   │
│  Le modèle peut créer un "super-neurone" qui détecte            │
│  parfaitement "Marseille" mais échoue sur "Bordeaux"            │
│                                                                 │
│  AVEC DROPOUT                                                   │
│  Ce super-neurone est parfois éteint !                          │
│  → Le modèle doit apprendre des patterns plus généraux          │
│  → Plusieurs neurones apprennent à détecter les villes          │
│  → Si l'un est éteint, les autres compensent                    │
└─────────────────────────────────────────────────────────────────┘

C'est comme entraîner une équipe de football : si le meilleur joueur est parfois sur le banc, les autres doivent progresser.

---
Exemple concret avec notre phrase

Phrase : "Je veux aller de Marseille à Paris"

Entraînement STEP 100 (dropout actif) :
┌─────────────────────────────────────────────────────────────────┐
│  Neurones du Tok2Vec (96 dimensions)                            │
│                                                                 │
│  [1][2][3][X][5][6][7][8][X][10]...[96]                         │
│         ↑              ↑                                        │
│      neurone 4      neurone 9                                   │
│       éteint         éteint                                     │
│                                                                 │
│  Le modèle doit prédire "Marseille = DÉPART"                    │
│  SANS l'aide des neurones 4 et 9                                │
└─────────────────────────────────────────────────────────────────┘

Entraînement STEP 101 (dropout différent) :
┌─────────────────────────────────────────────────────────────────┐
│  [1][X][3][4][5][X][7][8][9][10]...[96]                         │
│      ↑           ↑                                              │
│   neurone 2   neurone 6                                         │
│    éteint      éteint                                           │
│                                                                 │
│  Maintenant les neurones 4 et 9 sont actifs                     │
│  mais 2 et 6 sont éteints → configuration différente            │
└─────────────────────────────────────────────────────────────────┘

À chaque step, le modèle voit une "version appauvrie" de lui-même.

---
Différence Entraînement vs Inférence

Le dropout ne s'applique que pendant l'ENTRAÎNEMENT :

┌─────────────────────────────────────────────────────────────────┐
│  ENTRAÎNEMENT                                                   │
│  dropout = 0.1 → 10% des neurones éteints aléatoirement         │
│  + Scaling : sortie × (1 / 0.9) pour compenser                  │
│                                                                 │
│  INFÉRENCE (prédiction sur nouvelles phrases)                   │
│  dropout = 0 → TOUS les neurones actifs                         │
│  Pas de scaling nécessaire                                      │
└─────────────────────────────────────────────────────────────────┘

Le scaling est important : si 10% des neurones sont éteints, les 90% restants doivent "compenser" en multipliant leur sortie par 1/0.9 ≈ 1.11.

---
Pourquoi 10% (dropout = 0.1) ?

┌─────────────────────────────────────────────────────────────────┐
│  Valeur │  Effet                                                │
├─────────┼───────────────────────────────────────────────────────┤
│  0.0    │  Pas de régularisation → risque de sur-apprentissage  │
│  0.1    │  Léger → bon pour NER (tâche précise) ✓               │
│  0.3    │  Modéré → bon pour classification de texte            │
│  0.5    │  Fort → utilisé dans les gros modèles (BERT)          │
│  0.7+   │  Très fort → le modèle perd trop d'information        │
└─────────────────────────────────────────────────────────────────┘

Pour le NER, 10% est idéal car :
- La tâche nécessite de la PRÉCISION (identifier des mots exacts)
- Trop de dropout → le modèle "oublie" des indices importants
- Le dataset est assez grand (8000 phrases) → moins besoin de régularisation forte

---
Où s'applique le dropout ?

Dans notre architecture spaCy :

┌─────────────────────────────────────────────────────────────────┐
│  MultiHashEmbed                                                 │
│       ↓                                                         │
│  ─ ─ ─ ─ ─ ─ DROPOUT ici (entre les couches)                    │
│       ↓                                                         │
│  MaxoutWindowEncoder (couche 1)                                 │
│       ↓                                                         │
│  ─ ─ ─ ─ ─ ─ DROPOUT ici                                        │
│       ↓                                                         │
│  MaxoutWindowEncoder (couche 2)                                 │
│       ↓                                                         │
│  ─ ─ ─ ─ ─ ─ DROPOUT ici                                        │
│       ↓                                                         │
│  MaxoutWindowEncoder (couche 3)                                 │
│       ↓                                                         │
│  ─ ─ ─ ─ ─ ─ DROPOUT ici                                        │
│       ↓                                                         │
│  MaxoutWindowEncoder (couche 4)                                 │
│       ↓                                                         │
│  TransitionBasedParser                                          │
└─────────────────────────────────────────────────────────────────┘

Le dropout s'applique ENTRE les couches, pas à l'intérieur.

---
Configuration dans spaCy

[training]
dropout = 0.1  # 10% des neurones désactivés

C'est le seul paramètre ! spaCy gère automatiquement :
- L'application du dropout pendant l'entraînement
- La désactivation pendant l'inférence
- Le scaling pour compenser

---
Impact sur les performances

┌─────────────────────────────────────────────────────────────────┐
│  SANS DROPOUT (dropout = 0)                                     │
│                                                                 │
│  Train F1: 99.8%  │  Test F1: 89.2%                             │
│                   │                                             │
│  ↑ Grande différence = sur-apprentissage !                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  AVEC DROPOUT (dropout = 0.1)                                   │
│                                                                 │
│  Train F1: 96.5%  │  Test F1: 95.7%                             │
│                   │                                             │
│  ↑ Écart faible = bonne généralisation !                        │
└─────────────────────────────────────────────────────────────────┘

Le dropout "sacrifie" un peu de performance sur l'entraînement pour gagner sur les nouvelles données.

---
★ Insight ─────────────────────────────────────
Le dropout est une forme d'"ensemble learning" implicite. À chaque step, on entraîne un sous-réseau différent. À l'inférence, on utilise la moyenne de tous ces sous-réseaux (via le scaling).

Pourquoi pas de dropout sur l'embedding ? Les embeddings sont déjà régularisés par le hash (collisions). Ajouter du dropout sur les embeddings perdrait trop d'information lexicale.

Dropout + Early Stopping = combo gagnant. Le dropout ralentit légèrement l'apprentissage, ce qui laisse plus de temps à l'early stopping pour détecter le sur-apprentissage.
─────────────────────────────────────────────────
