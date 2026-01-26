⏺ L'Optimiseur Adam avec des exemples concrets

Prenons l'entraînement de notre modèle NER sur la phrase : "Je veux aller de Marseille à Paris"

---
Qu'est-ce qu'un optimiseur ?

Pendant l'entraînement, le modèle fait des erreurs. L'optimiseur ajuste les poids du réseau pour réduire ces erreurs.

┌─────────────────────────────────────────────────────────────────┐
│  CYCLE D'ENTRAÎNEMENT                                           │
│                                                                 │
│  1. Phrase → Modèle → Prédiction                                │
│     "de Marseille à Paris" → [OUT, ???, OUT, ???]               │
│                                                                 │
│  2. Comparaison avec la vérité                                  │
│     Vérité : [OUT, B-DEP, OUT, B-ARR]                           │
│     Erreur : le modèle a dit B-ARR au lieu de B-DEP             │
│                                                                 │
│  3. Calcul du gradient (direction de correction)                │
│     ∇L = "pour réduire l'erreur, augmente ce poids de 0.01..."  │
│                                                                 │
│  4. OPTIMISEUR ajuste les poids                                 │
│     poids_nouveau = poids_ancien - learning_rate × gradient     │
└─────────────────────────────────────────────────────────────────┘

---
Le problème de la descente de gradient simple (SGD)

La descente de gradient basique a des défauts :

┌─────────────────────────────────────────────────────────────────┐
│  PROBLÈME 1 : Oscillations                                      │
│                                                                 │
│  Erreur ↑                                                       │
│        │    /\    /\                                            │
│        │   /  \  /  \   ← Le modèle oscille autour du minimum   │
│        │  /    \/    \                                          │
│        │_/            \____                                     │
│        └───────────────────→ Itérations                         │
│                                                                 │
│  Cause : le gradient change brusquement de direction            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PROBLÈME 2 : Learning rate unique                              │
│                                                                 │
│  Certains paramètres sont souvent mis à jour (mots fréquents)   │
│  D'autres rarement (mots rares comme "Pontarlier")              │
│                                                                 │
│  Un learning rate unique ne convient pas à tous                 │
└─────────────────────────────────────────────────────────────────┘

---
Adam : la solution adaptative

Adam (Adaptive Moment Estimation) résout ces problèmes avec 2 mécanismes :

┌─────────────────────────────────────────────────────────────────┐
│  MÉCANISME 1 : Momentum (beta1 = 0.9)                           │
│                                                                 │
│  Au lieu de suivre uniquement le gradient actuel,               │
│  on garde une "mémoire" des gradients passés.                   │
│                                                                 │
│  m_t = 0.9 × m_{t-1} + 0.1 × gradient_actuel                    │
│         ↑                    ↑                                  │
│    90% du passé        10% du présent                           │
│                                                                 │
│  → Lisse les oscillations, comme une bille avec de l'inertie    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MÉCANISME 2 : Learning rate adaptatif (beta2 = 0.999)          │
│                                                                 │
│  On garde aussi la mémoire des gradients² (variance)            │
│                                                                 │
│  v_t = 0.999 × v_{t-1} + 0.001 × gradient²                      │
│                                                                 │
│  Puis on divise la mise à jour par √v_t :                       │
│  - Si le gradient varie beaucoup → petits pas (prudent)         │
│  - Si le gradient est stable → grands pas (confiant)            │
└─────────────────────────────────────────────────────────────────┘

---
Visualisation de l'effet du momentum

Sans momentum (SGD) :
┌─────────────────────────────────────────────────────────────────┐
│  Erreur ↑                                                       │
│        │    /\    /\                                            │
│        │   /  \  /  \  /\                                       │
│        │  /    \/    \/  \                                      │
│        │_/                \_____                                │
│        └───────────────────────→ Itérations                     │
│                                                                 │
│  ↑ Oscillations, convergence lente                              │
└─────────────────────────────────────────────────────────────────┘

Avec momentum (Adam) :
┌─────────────────────────────────────────────────────────────────┐
│  Erreur ↑                                                       │
│        │\                                                       │
│        │ \                                                      │
│        │  \                                                     │
│        │   \_____________                                       │
│        └───────────────────────→ Itérations                     │
│                                                                 │
│  ↑ Trajectoire lisse, convergence rapide                        │
└─────────────────────────────────────────────────────────────────┘

---
Le Warmup linéaire

Au début de l'entraînement, les poids sont aléatoires. Faire de grandes mises à jour serait risqué.

┌─────────────────────────────────────────────────────────────────┐
│  warmup_steps = 250                                             │
│  initial_rate = 0.00005 (5e-5)                                  │
│                                                                 │
│  Learning Rate                                                  │
│        ↑                                                        │
│  5e-5  │          /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\                        │
│        │         /                      \                       │
│        │        /                        \                      │
│        │       /                          \                     │
│    0   │______/                            \_______________     │
│        └────────────────────────────────────────────────────→   │
│        0      250                        20000          Steps   │
│              ↑                             ↑                    │
│          Warmup                         Decay                   │
└─────────────────────────────────────────────────────────────────┘

Phase 1 (0 → 250 steps) : Warmup
- Le learning rate monte de 0 à 5e-5
- Le modèle "s'échauffe" doucement
- Évite les mises à jour trop agressives au début

Phase 2 (250 → 20000 steps) : Decay linéaire
- Le learning rate descend progressivement vers 0
- Les ajustements deviennent de plus en plus fins
- Le modèle "peaufine" ses prédictions

---
Le Gradient Clipping

Parfois, le gradient "explose" (devient très grand). Le clipping le limite :

┌─────────────────────────────────────────────────────────────────┐
│  grad_clip = 1.0                                                │
│                                                                 │
│  Si ||gradient|| > 1.0 :                                        │
│     gradient = gradient × (1.0 / ||gradient||)                  │
│                                                                 │
│  Exemple :                                                      │
│  Gradient brut : [5.0, 3.0, 4.0]  (norme ≈ 7.07)                │
│  Gradient clippé : [0.71, 0.42, 0.57]  (norme = 1.0)            │
└─────────────────────────────────────────────────────────────────┘

Pourquoi c'est important ? Sans clipping, un gradient géant peut :
- Faire "sauter" le modèle hors de la zone de bonnes solutions
- Causer des NaN (Not a Number) qui plantent l'entraînement

---
Le Weight Decay (régularisation L2)

Pour éviter le sur-apprentissage, on pénalise les poids trop grands :

┌─────────────────────────────────────────────────────────────────┐
│  L2 = 0.01                                                      │
│                                                                 │
│  À chaque mise à jour :                                         │
│  poids_nouveau = poids_ancien × (1 - 0.01) - lr × gradient      │
│                               ↑                                 │
│              Réduction de 1% du poids                           │
│                                                                 │
│  Effet : les poids "oubliés" (non utilisés) rétrécissent        │
│          Les poids importants sont renforcés par le gradient    │
└─────────────────────────────────────────────────────────────────┘

C'est comme une taxe sur les poids : plus un poids est grand, plus il "paie". Seuls les poids vraiment utiles restent grands.

---
Configuration dans spaCy

Voici les paramètres utilisés dans notre projet :

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9              # Momentum du gradient
beta2 = 0.999            # Momentum du gradient²
L2 = 0.01                # Weight decay
grad_clip = 1.0          # Limite du gradient
eps = 0.00000001         # Évite division par zéro

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250       # Steps de montée
total_steps = 20000      # Durée totale
initial_rate = 0.00005   # Learning rate max (5e-5)

Tableau récapitulatif :
┌─────────────────────┬─────────────────────────────────────────────┐
│  Paramètre          │  Rôle                                       │
├─────────────────────┼─────────────────────────────────────────────┤
│ beta1 = 0.9         │ 90% momentum direction (lisse oscillations) │
│ beta2 = 0.999       │ 99.9% momentum vitesse (adapte le LR)       │
│ L2 = 0.01           │ Régularisation (évite sur-apprentissage)    │
│ grad_clip = 1.0     │ Sécurité (évite explosion des gradients)    │
│ warmup_steps = 250  │ Échauffement progressif                     │
│ initial_rate = 5e-5 │ Learning rate maximal                       │
└─────────────────────┴─────────────────────────────────────────────┘

---
Exemple concret d'une mise à jour

Imaginons un poids W qui influence la prédiction "Marseille = DÉPART" :

┌─────────────────────────────────────────────────────────────────┐
│  Step 100 (pendant le warmup)                                   │
│                                                                 │
│  W_actuel = 0.5                                                 │
│  gradient = 0.2 (le poids doit augmenter)                       │
│  learning_rate = 100/250 × 5e-5 = 2e-5 (warmup)                 │
│                                                                 │
│  Mise à jour Adam :                                             │
│  m = 0.9 × m_précédent + 0.1 × 0.2                              │
│  v = 0.999 × v_précédent + 0.001 × 0.2²                         │
│  W_nouveau = W × (1-0.01) - 2e-5 × m / √v                       │
│            = 0.495 - petite_correction                          │
│            ≈ 0.495 + 0.0001                                     │
│            = 0.4951                                             │
│                                                                 │
│  → Changement minuscule, le modèle avance prudemment            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 1000 (après le warmup)                                    │
│                                                                 │
│  learning_rate = 5e-5 (maximal)                                 │
│  m et v ont accumulé de l'information                           │
│                                                                 │
│  → Mises à jour plus significatives                             │
│  → Le modèle apprend activement                                 │
└─────────────────────────────────────────────────────────────────┘

---
★ Insight ─────────────────────────────────────
Pourquoi beta1 = 0.9 et beta2 = 0.999 ? Ce sont les valeurs par défaut du papier original d'Adam (2014). Elles fonctionnent bien pour la plupart des tâches NLP. beta2 plus grand que beta1 car la variance des gradients fluctue moins que leur direction.

Le warmup est crucial pour les transformers mais aussi utile pour les petits modèles. Sans warmup, le modèle peut "mémoriser" des patterns incorrects au début (quand les poids sont aléatoires) et avoir du mal à s'en défaire ensuite.

Learning rate = 5e-5 : C'est petit ! C'est normal pour le NLP où les embeddings sont sensibles. Un LR de 0.01 (classique en vision) ferait exploser les gradients.
─────────────────────────────────────────────────
