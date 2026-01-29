⏺ L'Early Stopping avec des exemples concrets

Prenons l'entraînement de notre modèle NER sur 8000 phrases.

---
Le problème : quand s'arrêter ?

L'entraînement pourrait continuer indéfiniment. Mais à un moment, le modèle commence à sur-apprendre :

┌─────────────────────────────────────────────────────────────────┐
│  Performance                                                    │
│        ↑                                                        │
│        │                    ╭──────── Train F1                  │
│        │                 ╭──╯                                   │
│        │              ╭──╯                                      │
│        │      ╭───────╯                                         │
│        │    ╭─╯     ╭─────────────── Test F1 (plateau)          │
│        │  ╭─╯   ╭───╯   ╲                                       │
│        │╭─╯ ╭───╯        ╲_______ Test F1 (déclin)              │
│        ├────────────────────────────────────────────→ Steps     │
│        0     1000    2000    3000    4000    5000               │
│                        ↑                                        │
│               Point optimal pour s'arrêter                      │
└─────────────────────────────────────────────────────────────────┘

Après le point optimal :
- Le Train F1 continue à monter (le modèle mémorise)
- Le Test F1 stagne puis descend (le modèle généralise moins bien)

---
La solution : Early Stopping

On surveille la performance sur un ensemble de VALIDATION (dev) et on s'arrête quand elle ne s'améliore plus.

┌─────────────────────────────────────────────────────────────────┐
│  patience = 1600                                                │
│  eval_frequency = 200                                           │
│                                                                 │
│  Signification :                                                │
│  → On évalue le modèle toutes les 200 steps                     │
│  → Si pas d'amélioration pendant 1600 évaluations → STOP        │
│                                                                 │
│  En steps : 1600 × 200 = 320,000 steps max de stagnation        │
│  (mais max_steps = 20,000 donc souvent atteint avant)           │
└─────────────────────────────────────────────────────────────────┘

---
Comment ça fonctionne

Voici le déroulement typique d'un entraînement :

┌─────────────────────────────────────────────────────────────────┐
│  Step 0    : Évaluation initiale, F1 = 0.00                     │
│  Step 200  : Évaluation, F1 = 0.45 ✓ (amélioration)             │
│              → Sauvegarde model-best, compteur = 0              │
│  Step 400  : Évaluation, F1 = 0.68 ✓ (amélioration)             │
│              → Sauvegarde model-best, compteur = 0              │
│  Step 600  : Évaluation, F1 = 0.82 ✓ (amélioration)             │
│              → Sauvegarde model-best, compteur = 0              │
│  Step 800  : Évaluation, F1 = 0.81 ✗ (pas d'amélioration)       │
│              → compteur = 1                                     │
│  Step 1000 : Évaluation, F1 = 0.83 ✓ (amélioration)             │
│              → Sauvegarde model-best, compteur = 0              │
│  ...                                                            │
│  Step 3000 : Évaluation, F1 = 0.957 (meilleur jusqu'ici)        │
│              → Sauvegarde model-best, compteur = 0              │
│  Step 3200 : F1 = 0.955 ✗ → compteur = 1                        │
│  Step 3400 : F1 = 0.956 ✗ → compteur = 2                        │
│  Step 3600 : F1 = 0.954 ✗ → compteur = 3                        │
│  ...                                                            │
│  (si compteur atteint 1600 → EARLY STOP)                        │
└─────────────────────────────────────────────────────────────────┘

---
Les deux modèles sauvegardés

spaCy sauvegarde DEUX versions du modèle :

┌─────────────────────────────────────────────────────────────────┐
│  models/spacy-ner/                                              │
│  ├── model-best/    ← Meilleur F1 jamais atteint                │
│  │   └── ...        (celui qu'on utilise en production)         │
│  └── model-last/    ← Dernier checkpoint                        │
│      └── ...        (utile pour reprendre l'entraînement)       │
└─────────────────────────────────────────────────────────────────┘

Pourquoi garder model-last ?
- Si on veut reprendre l'entraînement plus tard
- Pour analyser la différence entre "meilleur" et "dernier"
- Pour déboguer si le modèle a divergé

---
Visualisation du compteur de patience

┌─────────────────────────────────────────────────────────────────┐
│  F1 Score                                                       │
│        ↑                                                        │
│  0.96 ─┤               ★ Best = 0.957                           │
│        │              ╱ ╲                                       │
│  0.94 ─┤            ╱    ╲  ╱╲  ╱╲                               │
│        │          ╱        ╲╱  ╲╱  ╲                             │
│  0.92 ─┤        ╱                    ╲                          │
│        │      ╱                       ╲                         │
│  0.90 ─┤    ╱                                                   │
│        │  ╱                                                     │
│        ├────────────────────────────────────────────→ Steps     │
│                                                                 │
│  Compteur patience:                                             │
│        0  0  0  0  1  2  3  4  5  6  ...  1600 → STOP           │
│              ↑                                                  │
│         Dernier reset (nouveau best)                            │
└─────────────────────────────────────────────────────────────────┘

---
Pourquoi patience = 1600 ?

┌─────────────────────────────────────────────────────────────────┐
│  patience │  Comportement                                       │
├───────────┼─────────────────────────────────────────────────────┤
│    100    │  Trop impatient ! Peut s'arrêter dans une "vallée"  │
│           │  temporaire et rater un meilleur pic                │
│    500    │  Raisonnable pour des datasets petits               │
│   1600    │  Conservateur, laisse le temps aux fluctuations ✓   │
│   5000    │  Très patient, risque de gaspiller du temps         │
└───────────┴─────────────────────────────────────────────────────┘

Avec patience = 1600 et eval_frequency = 200 :
- Le modèle peut "stagner" pendant 1600 × 200 = 320,000 steps
- Mais max_steps = 20,000, donc on atteint rarement cette limite
- C'est une "assurance" plutôt qu'un critère d'arrêt principal

---
Interaction avec max_steps

Notre configuration :

[training]
patience = 1600
max_steps = 20000
eval_frequency = 200

Deux critères d'arrêt possibles :

┌─────────────────────────────────────────────────────────────────┐
│  CRITÈRE 1 : max_steps atteint                                  │
│  → On a fait 20,000 steps → STOP                                │
│  → C'est souvent ce qui arrive dans notre cas                   │
│                                                                 │
│  CRITÈRE 2 : patience épuisée                                   │
│  → Pas d'amélioration pendant 1600 évaluations → STOP           │
│  → Plus rare, indique que le modèle a convergé tôt              │
└─────────────────────────────────────────────────────────────────┘

---
Exemple réel de notre entraînement

Voici un extrait typique des logs spaCy :

┌─────────────────────────────────────────────────────────────────┐
│  E    #       LOSS NER    ENTS_F    ENTS_P    ENTS_R    SCORE   │
│  ---  ------  ----------  --------  --------  --------  ------  │
│  0    0       0.00        0.00      0.00      0.00      0.00    │
│  0    200     12.45       0.68      0.72      0.65      0.68    │
│  1    400     8.32        0.85      0.88      0.82      0.85    │
│  1    600     5.21        0.91      0.93      0.89      0.91    │
│  2    800     3.45        0.94      0.95      0.93      0.94    │
│  2    1000    2.67        0.95      0.96      0.94      0.95    │
│  3    1200    1.98        0.956     0.962     0.950     0.956   │
│  3    1400    1.45        0.957     0.963     0.951     0.957 ✓ │
│  4    1600    1.12        0.955     0.961     0.949     0.955   │
│  ...                                                            │
│  25   5000    0.23        0.957     0.964     0.950     0.957   │
│                                                                 │
│  ✓ = nouveau model-best sauvegardé                              │
└─────────────────────────────────────────────────────────────────┘

Observations :
- Le meilleur score (0.957) est atteint vers step 1400
- Après, le score oscille autour de cette valeur
- model-best est sauvegardé à step 1400

---
Configuration dans spaCy

[training]
patience = 1600         # Évaluations sans amélioration avant arrêt
max_steps = 20000       # Limite absolue de steps
eval_frequency = 200    # Fréquence d'évaluation (en steps)

[training.score_weights]
ents_f = 1.0            # Métrique surveillée pour early stopping

Le `ents_f = 1.0` signifie qu'on surveille le F1-score des entités. Si on avait plusieurs métriques, on pourrait les pondérer.

---
★ Insight ─────────────────────────────────────
Pourquoi eval_frequency = 200 ? C'est un compromis entre précision et vitesse. Évaluer trop souvent (50) ralentit l'entraînement. Évaluer trop rarement (1000) risque de rater le point optimal.

La patience "large" (1600) est intentionnelle. Le F1-score peut fluctuer de ±0.5% d'une évaluation à l'autre à cause du mini-batch sampling. Une patience courte s'arrêterait prématurément sur ces fluctuations.

model-best vs model-last : En production, utilisez TOUJOURS model-best. Le model-last peut avoir légèrement sur-appris. La différence est souvent de 0.1-0.5% de F1, mais sur 10,000 prédictions, ça fait 10-50 erreurs évitées.
─────────────────────────────────────────────────
