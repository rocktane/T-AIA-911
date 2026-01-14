⏺ Le MaxoutWindowEncoder avec des exemples concrets

Prenons la phrase : "Je veux aller de Marseille à Paris"

---
Le problème : un mot seul ne suffit pas

Après le MultiHashEmbed, chaque mot a son embedding. Mais regardons ces deux phrases :

Phrase A: "Je veux aller de Marseille à Paris"
Phrase B: "Je veux aller de Paris à Marseille"

Le mot "Paris" a le MÊME embedding dans les deux phrases. Pourtant :
- Phrase A → Paris = ARRIVÉE
- Phrase B → Paris = DÉPART

L'embedding seul ne capture pas le CONTEXTE. C'est le rôle de l'encodeur.

---
La fenêtre glissante (window)

L'encodeur regarde chaque mot avec ses voisins. Avec window_size = 1 :

┌─────────────────────────────────────────────────────────────────┐
│  Phrase: "Je veux aller de Marseille à Paris"                   │
│           0   1     2    3     4      5   6                     │
│                                                                 │
│  Pour analyser "Paris" (position 6) :                           │
│  Fenêtre = [à, Paris, ∅]                                        │
│             ↑    ↑    ↑                                         │
│          gauche mot  droite (rien après)                        │
└─────────────────────────────────────────────────────────────────┘

Le mot "à" juste avant "Paris" est un indice crucial : "à [ville]" = arrivée !

---
Les 4 couches de profondeur (depth = 4)

Une seule couche ne voit que les voisins directs. Mais avec 4 couches, l'information se PROPAGE :

COUCHE 1 — Chaque mot voit ±1 voisin
┌─────────────────────────────────────────────────────────────────┐
│  Position 6 "Paris" voit : ["à", "Paris", ∅]                    │
│  → Paris sait qu'il y a "à" juste avant                         │
│                                                                 │
│  Position 5 "à" voit : ["Marseille", "à", "Paris"]              │
│  → "à" sait qu'il y a "Marseille" avant et "Paris" après        │
└─────────────────────────────────────────────────────────────────┘

COUCHE 2 — L'information se propage
┌─────────────────────────────────────────────────────────────────┐
│  "Paris" reçoit l'information de "à"                            │
│  Or "à" connaissait "Marseille" depuis la couche 1              │
│  → Paris sait maintenant qu'il y a Marseille quelque part       │
└─────────────────────────────────────────────────────────────────┘

COUCHE 3 — Encore plus de contexte
┌─────────────────────────────────────────────────────────────────┐
│  "Paris" reçoit l'info de "Marseille"                           │
│  Or "Marseille" connaissait "de" depuis la couche 1             │
│  → Paris sait qu'il y a "de [ville]" avant lui                  │
└─────────────────────────────────────────────────────────────────┘

COUCHE 4 — Vision complète
┌─────────────────────────────────────────────────────────────────┐
│  "Paris" connaît toute la structure : "de X à Paris"            │
│  → Paris SAIT qu'il est l'arrivée (après "à")                   │
│  → Paris SAIT que Marseille est le départ (après "de")          │
└─────────────────────────────────────────────────────────────────┘

C'est l'effet "téléphone arabe" : à chaque couche, l'information des voisins se transmet.

---
Portée effective

Avec window_size = 1 et depth = 4, chaque mot "voit" jusqu'à ±4 positions :

┌─────────────────────────────────────────────────────────────────┐
│  Couche 1 : portée ±1                                           │
│  Couche 2 : portée ±2                                           │
│  Couche 3 : portée ±3                                           │
│  Couche 4 : portée ±4                                           │
│                                                                 │
│  Phrase: "Je veux aller de Marseille à Paris"                   │
│           ←────────── 7 mots ──────────→                        │
│                                                                 │
│  Après 4 couches, "Paris" a accès à toute la phrase !           │
└─────────────────────────────────────────────────────────────────┘

---
L'activation Maxout

À chaque couche, le réseau calcule plusieurs "candidats" et garde le maximum :

┌─────────────────────────────────────────────────────────────────┐
│  maxout_pieces = 3                                              │
│                                                                 │
│  Pour chaque neurone :                                          │
│  Candidat 1 : W1 × entrée + b1 = 0.45                           │
│  Candidat 2 : W2 × entrée + b2 = 0.82  ← Maximum !              │
│  Candidat 3 : W3 × entrée + b3 = 0.31                           │
│                                                                 │
│  Sortie = max(0.45, 0.82, 0.31) = 0.82                          │
└─────────────────────────────────────────────────────────────────┘

Pourquoi Maxout ?
- Plus expressif que ReLU (peut apprendre des fonctions complexes)
- Pas de "neurones morts" (toujours au moins 1 candidat actif)
- Idéal pour les petits réseaux comme le nôtre

---
Visualisation complète

Voici le flux pour "Marseille" dans notre phrase :

┌─────────────────────────────────────────────────────────────────┐
│  ENTRÉE: Embeddings de la phrase (96 dim par mot)               │
│                                                                 │
│  "Je"  "veux" "aller" "de" "Marseille" "à" "Paris"              │
│   ↓      ↓      ↓      ↓       ↓        ↓     ↓                 │
│  [96]  [96]   [96]   [96]    [96]     [96]  [96]                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  COUCHE 1 : Convolution window_size=1                           │
│                                                                 │
│  Pour "Marseille" : concat([de], [Marseille], [à])              │
│  = vecteur de 96×3 = 288 dimensions                             │
│  → Projection linéaire → 96×3 candidats                         │
│  → Maxout → 96 dimensions                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  COUCHE 2, 3, 4 : Même processus                                │
│                                                                 │
│  À chaque couche, le contexte s'élargit                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  SORTIE: Vecteur contextualisé de 96 dim pour chaque mot        │
│                                                                 │
│  "Marseille" sait maintenant :                                  │
│  - Il est précédé de "de" (indicateur de départ)                │
│  - Il est suivi de "à Paris" (la destination)                   │
│  - Il est dans une structure "de X à Y"                         │
└─────────────────────────────────────────────────────────────────┘

---
Configuration dans spaCy

Voici les paramètres utilisés dans notre projet :

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96          # Même dimension que l'embedding
depth = 4           # 4 couches de convolution
window_size = 1     # Regarde ±1 mot à chaque couche
maxout_pieces = 3   # 3 candidats pour le maxout

Explication des paramètres :
┌─────────────────────┬─────────────────────────────────────────────┐
│  Paramètre          │  Description                                │
├─────────────────────┼─────────────────────────────────────────────┤
│ width = 96          │ Dimension des vecteurs (cohérent avec embed)│
│ depth = 4           │ Nombre de couches (portée ±4 mots)          │
│ window_size = 1     │ Voisins directs seulement par couche        │
│ maxout_pieces = 3   │ 3 candidats par activation                  │
└─────────────────────┴─────────────────────────────────────────────┘

---
Comparaison avec les Transformers

┌─────────────────────┬─────────────────────┬─────────────────────┐
│                     │  MaxoutWindowEncoder│    Transformer      │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Attention           │ Locale (±4 mots)    │ Globale (tout)      │
│ Complexité          │ O(n × d × w)        │ O(n² × d)           │
│ Mémoire             │ ~5 Mo               │ ~500 Mo             │
│ Vitesse             │ Très rapide         │ Lent (sans GPU)     │
│ Phrases longues     │ Limité              │ Excellent           │
│ Notre cas (7 mots)  │ Parfait             │ Surdimensionné      │
└─────────────────────┴─────────────────────┴─────────────────────┘

Pour nos phrases courtes (moyenne de 8 mots), le MaxoutWindowEncoder est idéal :
- Rapide : pas besoin de GPU
- Suffisant : 4 couches couvrent toute la phrase
- Léger : modèle de ~10 Mo total

---
Exemple avec une phrase plus longue

Phrase : "Demain matin je voudrais partir de la gare de Lyon à Marseille"
          0      1     2     3       4     5  6   7   8   9  10    11

Avec 12 mots, certains sont à distance > 4. Mais :

┌─────────────────────────────────────────────────────────────────┐
│  "Lyon" (position 9) après 4 couches voit : positions 5 à 11    │
│  → Voit "partir de la gare de Lyon à Marseille"                 │
│  → Suffisant pour comprendre le contexte !                      │
│                                                                 │
│  Les mots du début ("Demain matin je") sont moins importants    │
│  pour déterminer si Lyon est départ ou arrivée.                 │
└─────────────────────────────────────────────────────────────────┘

---
★ Insight ─────────────────────────────────────
Pourquoi depth = 4 ? C'est le minimum pour couvrir nos phrases typiques. Avec 3 couches (portée ±3), on risquerait de manquer le "de" ou "à" dans les phrases de 7+ mots. Avec 5+ couches, on gaspille des paramètres sans gain.

Maxout vs ReLU : ReLU peut "tuer" des neurones (sortie = 0 si entrée < 0). Maxout garde toujours le meilleur candidat parmi 3, donc pas de neurone mort. C'est crucial quand on a peu de couches.

Le window_size = 1 est un choix délibéré : en combinant avec depth = 4, on obtient une portée de ±4 mots tout en gardant le modèle léger. Un window_size = 2 doublerait les paramètres pour une portée similaire.
─────────────────────────────────────────────────
