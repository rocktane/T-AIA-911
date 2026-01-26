⏺ Le TransitionBasedParser avec des exemples concrets

Prenons la phrase : "Je veux aller de Marseille à Paris"

---
Le problème : identifier les entités

Après le Tok2Vec, chaque mot a un vecteur contextualisé de 96 dimensions. Maintenant, il faut décider : quel mot est un DÉPART ? quel mot est une ARRIVÉE ?

C'est le rôle du parser NER (Named Entity Recognition).

---
L'approche "shift-reduce"

Le TransitionBasedParser traite la phrase de GAUCHE à DROITE, mot par mot, en prenant des décisions séquentielles :

┌─────────────────────────────────────────────────────────────────┐
│  Phrase: "Je veux aller de Marseille à Paris"                   │
│                                                                 │
│  Étape 1: [Je]     → Décision: OUT (pas une entité)             │
│  Étape 2: [veux]   → Décision: OUT                              │
│  Étape 3: [aller]  → Décision: OUT                              │
│  Étape 4: [de]     → Décision: OUT                              │
│  Étape 5: [Marseille] → Décision: BEGIN-DEPART (début d'entité) │
│  Étape 6: [à]      → Décision: OUT                              │
│  Étape 7: [Paris]  → Décision: BEGIN-ARRIVEE (début d'entité)   │
└─────────────────────────────────────────────────────────────────┘

Le parser "avance" dans la phrase comme un curseur, et à chaque position il décide quelle ACTION effectuer.

---
Les 4 actions possibles

┌─────────┬────────────────────────────────────────────────────────┐
│ Action  │ Signification                                          │
├─────────┼────────────────────────────────────────────────────────┤
│ OUT     │ Ce mot n'est PAS une entité                            │
│ BEGIN   │ Ce mot est le DÉBUT d'une entité                       │
│ IN      │ Ce mot est AU MILIEU d'une entité (multi-mots)         │
│ LAST    │ Ce mot est la FIN d'une entité (multi-mots)            │
└─────────┴────────────────────────────────────────────────────────┘

Avec nos 2 types d'entités (DEPART, ARRIVEE), on a en réalité :
- OUT (1 action)
- BEGIN-DEPART, BEGIN-ARRIVEE (2 actions)
- IN-DEPART, IN-ARRIVEE (2 actions)
- LAST-DEPART, LAST-ARRIVEE (2 actions)

Total : 7 actions possibles.

---
Exemple avec une entité multi-mots

Phrase : "Je pars de Saint-Pierre-des-Corps à Lyon"

┌─────────────────────────────────────────────────────────────────┐
│  [Je]              → OUT                                        │
│  [pars]            → OUT                                        │
│  [de]              → OUT                                        │
│  [Saint]           → BEGIN-DEPART  ← Début de l'entité          │
│  [-]               → IN-DEPART     ← Continuation               │
│  [Pierre]          → IN-DEPART     ← Continuation               │
│  [-]               → IN-DEPART     ← Continuation               │
│  [des]             → IN-DEPART     ← Continuation               │
│  [-]               → IN-DEPART     ← Continuation               │
│  [Corps]           → LAST-DEPART   ← Fin de l'entité            │
│  [à]               → OUT                                        │
│  [Lyon]            → BEGIN-ARRIVEE (et LAST car 1 seul mot)     │
└─────────────────────────────────────────────────────────────────┘

Note : Pour une entité d'un seul mot (comme "Lyon"), on utilise directement BEGIN (qui fait aussi office de LAST).

---
Le mécanisme de décision

À chaque position, le parser utilise un réseau de neurones pour choisir l'action :

┌─────────────────────────────────────────────────────────────────┐
│  ENTRÉE                                                         │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Vecteur contextualisé du mot courant (96 dim)               ││
│  │ + État interne du parser (entités en cours)                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Couche cachée (hidden_width = 64 neurones)                  ││
│  │ Activation Maxout (maxout_pieces = 2)                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                    │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Couche de sortie (7 actions possibles)                      ││
│  │ Softmax → probabilités                                      ││
│  └─────────────────────────────────────────────────────────────┘│
│                            ↓                                    │
│  SORTIE : Action avec la probabilité la plus élevée             │
│  Exemple : BEGIN-DEPART (0.89), OUT (0.08), ...                 │
└─────────────────────────────────────────────────────────────────┘

---
Visualisation pas à pas

Phrase : "Je veux aller de Marseille à Paris"

Position 0 : "Je"
┌─────────────────────────────────────────────────────────────────┐
│  Vecteur de "Je" (contextualisé par Tok2Vec)                    │
│  → hidden layer → softmax                                       │
│                                                                 │
│  Probabilités :                                                 │
│  OUT: 0.98  BEGIN-DEP: 0.01  BEGIN-ARR: 0.01                    │
│                                                                 │
│  Décision : OUT ✓                                               │
└─────────────────────────────────────────────────────────────────┘

Position 4 : "Marseille"
┌─────────────────────────────────────────────────────────────────┐
│  Vecteur de "Marseille" (sait qu'il y a "de" avant)             │
│  → hidden layer → softmax                                       │
│                                                                 │
│  Probabilités :                                                 │
│  OUT: 0.02  BEGIN-DEP: 0.95  BEGIN-ARR: 0.03                    │
│        ↑                                                        │
│  Le "de" avant indique fortement un DÉPART                      │
│                                                                 │
│  Décision : BEGIN-DEPART ✓                                      │
└─────────────────────────────────────────────────────────────────┘

Position 6 : "Paris"
┌─────────────────────────────────────────────────────────────────┐
│  Vecteur de "Paris" (sait qu'il y a "à" avant)                  │
│  → hidden layer → softmax                                       │
│                                                                 │
│  Probabilités :                                                 │
│  OUT: 0.01  BEGIN-DEP: 0.02  BEGIN-ARR: 0.97                    │
│                                       ↑                         │
│  Le "à" avant indique fortement une ARRIVÉE                     │
│                                                                 │
│  Décision : BEGIN-ARRIVEE ✓                                     │
└─────────────────────────────────────────────────────────────────┘

---
L'Oracle Training

Pendant l'entraînement, on utilise un "oracle" qui connaît les bonnes réponses :

┌─────────────────────────────────────────────────────────────────┐
│  ENTRAÎNEMENT                                                   │
│                                                                 │
│  Phrase : "de Marseille à Paris"                                │
│  Labels : [OUT, DEPART, OUT, ARRIVEE]                           │
│                                                                 │
│  Position 1 "Marseille" :                                       │
│  - Prédiction du modèle : BEGIN-ARRIVEE (erreur !)              │
│  - Oracle dit : BEGIN-DEPART                                    │
│  - → Ajuste les poids pour favoriser BEGIN-DEPART               │
│                                                                 │
│  update_with_oracle_cut_size = 100                              │
│  → Corrige les 100 premières erreurs par batch                  │
└─────────────────────────────────────────────────────────────────┘

---
Configuration dans spaCy

Voici les paramètres utilisés dans notre projet :

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
hidden_width = 64           # Neurones dans la couche cachée
maxout_pieces = 2           # Activation maxout
use_upper = true            # Utilise des features supplémentaires
update_with_oracle_cut_size = 100  # Corrections par batch

Explication des paramètres :
┌─────────────────────────────┬───────────────────────────────────┐
│  Paramètre                  │  Description                      │
├─────────────────────────────┼───────────────────────────────────┤
│ state_type = "ner"          │ Mode reconnaissance d'entités     │
│ hidden_width = 64           │ Taille de la couche de décision   │
│ maxout_pieces = 2           │ 2 candidats par activation        │
│ use_upper = true            │ Features du contexte global       │
│ update_with_oracle_cut_size │ Limite les corrections par batch  │
└─────────────────────────────┴───────────────────────────────────┘

---
Avantages du TransitionBasedParser

┌─────────────────────────────────────────────────────────────────┐
│  1. RAPIDITÉ                                                    │
│     Traitement en O(n) - une passe de gauche à droite           │
│     Idéal pour l'inférence en temps réel                        │
│                                                                 │
│  2. EFFICACITÉ MÉMOIRE                                          │
│     Pas besoin de stocker toutes les paires de mots             │
│     Contrairement à un CRF ou un Transformer                    │
│                                                                 │
│  3. DÉCISIONS EXPLICITES                                        │
│     On peut voir quelle action a été choisie à chaque étape     │
│     Facilite le débogage                                        │
└─────────────────────────────────────────────────────────────────┘

---
Limites du TransitionBasedParser

┌─────────────────────────────────────────────────────────────────┐
│  1. PAS DE CONTEXTE FUTUR                                       │
│     Le parser ne peut pas "regarder en avant"                   │
│     Quand il traite "Marseille", il ne sait pas encore          │
│     qu'il y a "Paris" après                                     │
│                                                                 │
│  2. ERREURS EN CASCADE                                          │
│     Si le parser se trompe sur un mot, les décisions            │
│     suivantes peuvent être affectées                            │
│                                                                 │
│  3. LIMITÉ POUR LES ENTITÉS IMBRIQUÉES                          │
│     Ne gère pas bien "Gare de Lyon à Paris" où                  │
│     "Lyon" fait partie de "Gare de Lyon"                        │
└─────────────────────────────────────────────────────────────────┘

Heureusement, le Tok2Vec compense partiellement ces limites en donnant au parser un contexte déjà enrichi par les 4 couches de convolution.

---
Comparaison avec CamemBERT

┌─────────────────────┬─────────────────────┬─────────────────────┐
│                     │ TransitionBasedParser│    CamemBERT        │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Direction           │ Gauche → Droite     │ Bidirectionnel      │
│ Contexte futur      │ Non (via Tok2Vec)   │ Oui (attention)     │
│ Vitesse             │ ~10,000 phrases/sec │ ~100 phrases/sec    │
│ Taille modèle       │ ~10 Mo              │ ~400 Mo             │
│ F1 sur notre tâche  │ 95.72%              │ 99.52%              │
│ GPU requis          │ Non                 │ Recommandé          │
└─────────────────────┴─────────────────────┴─────────────────────┘

Pour notre tâche simple (2 entités, phrases courtes), le TransitionBasedParser offre un excellent compromis vitesse/précision.

---
Exemple de cas difficile

Phrase : "À Marseille depuis Lyon"

Ici, l'ARRIVÉE vient AVANT le DÉPART dans la phrase. Le parser doit gérer :

┌─────────────────────────────────────────────────────────────────┐
│  [À]           → OUT                                            │
│  [Marseille]   → BEGIN-ARRIVEE  ← "À [ville]" = arrivée         │
│  [depuis]      → OUT                                            │
│  [Lyon]        → BEGIN-DEPART   ← "depuis [ville]" = départ     │
└─────────────────────────────────────────────────────────────────┘

Le parser réussit car le Tok2Vec lui donne le contexte : même si le parser lit de gauche à droite, chaque mot a déjà "vu" ses voisins grâce aux 4 couches de convolution.

---
★ Insight ─────────────────────────────────────
Pourquoi hidden_width = 64 ? C'est suffisant pour distinguer 7 actions. Plus grand (128, 256) ralentirait l'inférence sans gain de précision significatif pour notre tâche simple.

Le paradigme shift-reduce vient de l'analyse syntaxique (parsing de grammaires). spaCy l'adapte pour le NER car les entités peuvent être vues comme des "constituants" à reconnaître.

L'avantage caché : le TransitionBasedParser permet d'ajouter des contraintes. Par exemple, on peut interdire "IN-DEPART" juste après "BEGIN-ARRIVEE" (impossible d'avoir deux entités différentes collées). Ces contraintes améliorent la cohérence des prédictions.
─────────────────────────────────────────────────
