⏺ Le F1-Score avec des exemples concrets

Prenons l'évaluation de notre modèle NER sur la phrase : "Je veux aller de Marseille à Paris"

---
Le problème : évaluer un modèle NER

Comment mesurer si notre modèle est "bon" ? On pourrait compter le pourcentage de prédictions correctes (accuracy), mais c'est trompeur pour le NER.

┌─────────────────────────────────────────────────────────────────┐
│  Phrase : "Je veux aller de Marseille à Paris"                  │
│  Labels : [OUT, OUT, OUT, OUT, B-DEP, OUT, B-ARR]               │
│                                                                 │
│  7 mots, dont 5 sont OUT (71%)                                  │
│                                                                 │
│  Un modèle "stupide" qui prédit toujours OUT :                  │
│  Prédiction : [OUT, OUT, OUT, OUT, OUT, OUT, OUT]               │
│  Accuracy = 5/7 = 71%  ← Semble bon !                           │
│                                                                 │
│  Mais il rate TOUS les départs et arrivées !                    │
└─────────────────────────────────────────────────────────────────┘

L'accuracy est biaisée par la classe majoritaire (OUT).

---
Précision et Recall

Pour évaluer correctement, on utilise deux métriques complémentaires :

┌─────────────────────────────────────────────────────────────────┐
│  PRÉCISION (Precision)                                          │
│  "Parmi les entités que j'ai prédites, combien sont correctes?"│
│                                                                 │
│  Précision = Vrais Positifs / (Vrais Positifs + Faux Positifs)  │
│                                                                 │
│  Exemple :                                                      │
│  Prédiction : [Marseille=DEP, Paris=ARR, Lyon=DEP]              │
│  Vérité :     [Marseille=DEP, Paris=ARR]                        │
│                                                                 │
│  → Vrais Positifs = 2 (Marseille, Paris corrects)               │
│  → Faux Positifs = 1 (Lyon prédit mais pas une entité)          │
│  → Précision = 2 / 3 = 66.7%                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RECALL (Rappel)                                                │
│  "Parmi les vraies entités, combien ai-je trouvées ?"           │
│                                                                 │
│  Recall = Vrais Positifs / (Vrais Positifs + Faux Négatifs)     │
│                                                                 │
│  Exemple :                                                      │
│  Prédiction : [Marseille=DEP]                                   │
│  Vérité :     [Marseille=DEP, Paris=ARR]                        │
│                                                                 │
│  → Vrais Positifs = 1 (Marseille trouvé)                        │
│  → Faux Négatifs = 1 (Paris manqué)                             │
│  → Recall = 1 / 2 = 50%                                         │
└─────────────────────────────────────────────────────────────────┘

---
Le compromis Précision / Recall

Ces deux métriques sont souvent en tension :

┌─────────────────────────────────────────────────────────────────┐
│  Modèle "prudent" (peu de prédictions)                          │
│  → Haute précision (les rares prédictions sont correctes)       │
│  → Faible recall (beaucoup d'entités manquées)                  │
│                                                                 │
│  Modèle "généreux" (beaucoup de prédictions)                    │
│  → Faible précision (beaucoup de faux positifs)                 │
│  → Haut recall (trouve presque toutes les entités)              │
└─────────────────────────────────────────────────────────────────┘

Comment choisir ? On veut un ÉQUILIBRE.

---
Le F1-Score : la moyenne harmonique

Le F1 combine précision et recall de manière équilibrée :

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           F1 = 2 × (Précision × Recall)                         │
│               ─────────────────────────                         │
│                 Précision + Recall                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Pourquoi la moyenne HARMONIQUE et pas arithmétique ?

┌─────────────────────────────────────────────────────────────────┐
│  Exemple : Précision = 100%, Recall = 10%                       │
│                                                                 │
│  Moyenne arithmétique : (100 + 10) / 2 = 55%  ← Semble OK       │
│  Moyenne harmonique :   2×(1×0.1)/(1+0.1) = 18%  ← Punit le 10% │
│                                                                 │
│  La moyenne harmonique PÉNALISE les extrêmes.                   │
│  Un bon F1 nécessite que BOTH précision ET recall soient bons.  │
└─────────────────────────────────────────────────────────────────┘

---
Exemple concret avec notre modèle

Évaluons sur 10 phrases :

┌─────────────────────────────────────────────────────────────────┐
│  Vérité terrain :                                               │
│  - 10 DÉPARTS (villes de départ)                                │
│  - 10 ARRIVÉES (villes d'arrivée)                               │
│                                                                 │
│  Prédictions du modèle :                                        │
│  - 9 DÉPARTS corrects, 1 manqué                                 │
│  - 1 faux DÉPART (mot normal prédit comme ville)                │
│  - 10 ARRIVÉES correctes                                        │
└─────────────────────────────────────────────────────────────────┘

Calculs pour DÉPART :
┌─────────────────────────────────────────────────────────────────┐
│  Vrais Positifs (VP) = 9  (départs correctement identifiés)     │
│  Faux Positifs (FP) = 1   (mot prédit départ mais ne l'est pas) │
│  Faux Négatifs (FN) = 1   (vrai départ non détecté)             │
│                                                                 │
│  Précision = VP / (VP + FP) = 9 / 10 = 90%                      │
│  Recall = VP / (VP + FN) = 9 / 10 = 90%                         │
│  F1 = 2 × (0.9 × 0.9) / (0.9 + 0.9) = 90%                       │
└─────────────────────────────────────────────────────────────────┘

Calculs pour ARRIVÉE :
┌─────────────────────────────────────────────────────────────────┐
│  VP = 10, FP = 0, FN = 0                                        │
│                                                                 │
│  Précision = 10 / 10 = 100%                                     │
│  Recall = 10 / 10 = 100%                                        │
│  F1 = 100%                                                      │
└─────────────────────────────────────────────────────────────────┘

---
Micro vs Macro F1

Deux façons d'agréger les F1 par type d'entité :

┌─────────────────────────────────────────────────────────────────┐
│  MICRO F1 (utilisé par spaCy par défaut)                        │
│                                                                 │
│  On agrège TOUS les VP, FP, FN avant de calculer :              │
│  VP total = 9 + 10 = 19                                         │
│  FP total = 1 + 0 = 1                                           │
│  FN total = 1 + 0 = 1                                           │
│                                                                 │
│  Précision = 19 / 20 = 95%                                      │
│  Recall = 19 / 20 = 95%                                         │
│  Micro F1 = 95%                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  MACRO F1                                                       │
│                                                                 │
│  On calcule le F1 par type, puis on fait la moyenne :           │
│  F1(DÉPART) = 90%                                               │
│  F1(ARRIVÉE) = 100%                                             │
│                                                                 │
│  Macro F1 = (90 + 100) / 2 = 95%                                │
└─────────────────────────────────────────────────────────────────┘

Dans notre cas, les deux sont égaux. Mais si les types étaient déséquilibrés :
- Micro F1 → favorise les types fréquents
- Macro F1 → traite tous les types également

---
Configuration dans spaCy

[training.score_weights]
ents_f = 1.0    # F1-score global (poids 1.0)
ents_p = 0.0    # Précision seule (poids 0)
ents_r = 0.0    # Recall seul (poids 0)

Signification :
- On optimise UNIQUEMENT le F1 (ents_f = 1.0)
- On ne pondère pas séparément précision ou recall
- C'est le choix standard pour le NER

---
Interprétation des scores

┌─────────────────────────────────────────────────────────────────┐
│  F1 Score  │  Interprétation                                    │
├────────────┼────────────────────────────────────────────────────┤
│  < 50%     │  Mauvais, le modèle fait beaucoup d'erreurs        │
│  50-70%    │  Passable, utilisable avec prudence                │
│  70-85%    │  Bon, utilisable en production avec supervision    │
│  85-95%    │  Très bon, fiable pour la plupart des cas          │
│  > 95%     │  Excellent, proche de la performance humaine       │
└────────────┴────────────────────────────────────────────────────┘

Notre modèle spaCy : 95.72% F1 → Très bon !
CamemBERT : 99.52% F1 → Excellent !

---
Exact Match vs F1 par token

Pour le NER, on peut évaluer de deux façons :

┌─────────────────────────────────────────────────────────────────┐
│  F1 PAR TOKEN (ce que spaCy utilise)                            │
│                                                                 │
│  Vérité :     [B-DEP, I-DEP, I-DEP, OUT, B-ARR]                 │
│  Prédiction : [B-DEP, I-DEP, OUT, OUT, B-ARR]                   │
│                        ↑                                        │
│               Erreur sur 1 token                                │
│                                                                 │
│  → 4/5 tokens corrects = 80% token-level accuracy               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  EXACT MATCH (plus strict)                                      │
│                                                                 │
│  L'entité entière doit être correcte.                           │
│  "Saint-Pierre-des-Corps" → tous les tokens doivent être bons   │
│                                                                 │
│  Si un seul token est faux → entité considérée comme ratée      │
│                                                                 │
│  → Plus difficile, mais reflète mieux l'usage réel              │
└─────────────────────────────────────────────────────────────────┘

---
Matrice de confusion pour le NER

┌─────────────────────────────────────────────────────────────────┐
│                    PRÉDICTION                                   │
│              ┌─────────┬─────────┬─────────┐                    │
│              │  DÉPART │ ARRIVÉE │   OUT   │                    │
│  ┌───────────┼─────────┼─────────┼─────────┤                    │
│  │  DÉPART   │   VP    │   FP    │   FN    │                    │
│ V│           │  (9)    │   (0)   │   (1)   │                    │
│ É├───────────┼─────────┼─────────┼─────────┤                    │
│ R│ ARRIVÉE   │   FP    │   VP    │   FN    │                    │
│ I│           │   (0)   │  (10)   │   (0)   │                    │
│ T├───────────┼─────────┼─────────┼─────────┤                    │
│ É│   OUT     │   FP    │   FP    │   VN    │                    │
│  │           │   (1)   │   (0)   │  (50)   │                    │
│  └───────────┴─────────┴─────────┴─────────┘                    │
│                                                                 │
│  VP = Vrai Positif, FP = Faux Positif                           │
│  FN = Faux Négatif, VN = Vrai Négatif                           │
└─────────────────────────────────────────────────────────────────┘

---
Nos résultats

Voici les métriques de notre modèle spaCy entraîné :

┌─────────────────────────────────────────────────────────────────┐
│  Entité    │ Précision │  Recall  │    F1    │                  │
├────────────┼───────────┼──────────┼──────────┤                  │
│  DÉPART    │  99.62%   │  92.12%  │  95.72%  │                  │
│  ARRIVÉE   │ 100.00%   │  90.15%  │  94.82%  │                  │
├────────────┼───────────┼──────────┼──────────┤                  │
│  GLOBAL    │  99.80%   │  91.14%  │  95.57%  │  ← Micro F1      │
└────────────┴───────────┴──────────┴──────────┘                  │
│                                                                 │
│  Observation : Haute précision, recall légèrement plus bas      │
│  → Le modèle est "prudent", rate quelques entités               │
│  → Préférable à l'inverse (beaucoup de faux positifs)           │
└─────────────────────────────────────────────────────────────────┘

---
★ Insight ─────────────────────────────────────
Pourquoi optimiser le F1 et pas l'accuracy ? Pour le NER, les tokens OUT sont majoritaires (souvent 80%+). Un modèle qui prédit tout OUT aurait une haute accuracy mais un F1 de 0%. Le F1 force le modèle à trouver les entités rares.

Précision vs Recall pour notre cas : Une haute précision (99.62%) signifie que quand le modèle dit "c'est une ville", il a presque toujours raison. Un recall de 92% signifie qu'il manque 8% des villes. Pour une application de réservation de train, mieux vaut demander confirmation à l'utilisateur que d'envoyer un train vers la mauvaise ville !

Le F1 de 95.72% signifie que sur 1000 entités, le modèle en identifie correctement ~957. Les 43 erreurs se répartissent entre faux positifs et faux négatifs.
─────────────────────────────────────────────────
