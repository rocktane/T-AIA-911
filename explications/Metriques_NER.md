# Metriques d'evaluation pour le NER

Ce document explique les metriques utilisees pour evaluer la qualite d'un modele de Named Entity Recognition (NER), en particulier dans le contexte de l'extraction de villes de depart et d'arrivee.

---

## 1. La Matrice de Confusion

Avant de comprendre les metriques, il faut comprendre les quatre types de resultats possibles lors d'une prediction :

```
                    Realite
                ENTITE    NON-ENTITE
             ┌──────────┬──────────┐
   Predit    │    TP    │    FP    │
   ENTITE    │  (Vrai   │  (Faux   │
             │ Positif) │ Positif) │
             ├──────────┼──────────┤
   Predit    │    FN    │    TN    │
 NON-ENTITE  │  (Faux   │  (Vrai   │
             │ Negatif) │ Negatif) │
             └──────────┴──────────┘
```

| Terme | Definition | Exemple NER |
|-------|------------|-------------|
| **TP** (True Positive) | Le modele predit une entite qui existe reellement | "Paris" predit comme DEPART, et c'est bien un depart |
| **FP** (False Positive) | Le modele predit une entite qui n'existe pas | "aujourd'hui" predit comme DEPART (erreur) |
| **FN** (False Negative) | Le modele rate une entite qui existe | "Lyon" est un depart mais le modele ne l'a pas detecte |
| **TN** (True Negative) | Le modele ne predit rien et il n'y a rien | "le train" n'est pas une entite, le modele n'en predit pas |

> **Note** : En NER, les TN sont tres nombreux (la plupart des tokens ne sont pas des entites) et donc peu informatifs. C'est pourquoi les metriques NER se concentrent sur TP, FP et FN.

---

## 2. Precision

**Question** : *Parmi mes predictions, combien sont correctes ?*

```
                    TP
Precision = ─────────────
              TP + FP
```

### Interpretation

- **Precision haute** (proche de 1) : Quand le modele dit "c'est un DEPART", il a presque toujours raison
- **Precision basse** : Le modele fait beaucoup de fausses alertes (predit des entites qui n'en sont pas)

### Exemple concret

Le modele predit 100 entites DEPART :
- 95 sont des vrais departs (TP = 95)
- 5 sont des erreurs (FP = 5)

**Precision = 95 / (95 + 5) = 0.95 = 95%**

### Quand privilegier la precision ?

Quand les faux positifs sont couteux. Dans notre cas, un faux depart pourrait generer un itineraire incorrect.

---

## 3. Rappel (Recall)

**Question** : *Parmi les vraies entites, combien ai-je trouvees ?*

```
                  TP
Rappel = ─────────────
           TP + FN
```

### Interpretation

- **Rappel haut** (proche de 1) : Le modele trouve presque toutes les entites existantes
- **Rappel bas** : Le modele rate beaucoup d'entites (trop conservateur)

### Exemple concret

Il y a 100 vraies entites DEPART dans le corpus :
- Le modele en detecte 92 (TP = 92)
- Il en rate 8 (FN = 8)

**Rappel = 92 / (92 + 8) = 0.92 = 92%**

### Quand privilegier le rappel ?

Quand les faux negatifs sont couteux. Dans notre cas, rater un depart signifie que la phrase sera marquee INVALID.

---

## 4. F1-Score

**Question** : *Comment combiner precision et rappel en une seule metrique ?*

```
              2 × Precision × Rappel
F1-Score = ─────────────────────────────
              Precision + Rappel
```

### Pourquoi la moyenne harmonique et pas arithmetique ?

La moyenne arithmetique serait trop indulgente avec les modeles desequilibres :

| Scenario | Precision | Rappel | Moyenne Arithmetique | Moyenne Harmonique (F1) |
|----------|-----------|--------|---------------------|-------------------------|
| Equilibre | 90% | 90% | 90% | 90% |
| Desequilibre | 100% | 50% | 75% | **66.7%** |
| Desequilibre | 50% | 100% | 75% | **66.7%** |

La moyenne harmonique **penalise les valeurs extremes**. Un modele qui predit tout (rappel=100%) mais avec beaucoup d'erreurs (precision=50%) obtient un F1 de 66.7% et non 75%.

### Propriete mathematique

La moyenne harmonique est toujours <= a la moyenne arithmetique, et l'ecart est d'autant plus grand que les valeurs sont differentes.

```
Pour P=100% et R=50% :
- Arithmetique : (100 + 50) / 2 = 75%
- Harmonique : 2 × 100 × 50 / (100 + 50) = 66.7%
```

### Exemple dans le projet

D'apres le CLAUDE.md, voici les resultats du modele spaCy custom :

| Entite | Precision | Rappel | F1-Score |
|--------|-----------|--------|----------|
| DEPART | 99.62% | 92.12% | 95.72% |
| ARRIVEE | 100.00% | 90.15% | 94.82% |

Le F1 pour DEPART n'est pas (99.62 + 92.12) / 2 = 95.87%, mais bien la moyenne harmonique : **95.72%**

---

## 5. F1 Macro vs F1 Micro

### F1 Macro

Calcule le F1 pour chaque classe separement, puis fait la moyenne :

```
F1_macro = (F1_DEPART + F1_ARRIVEE) / 2
```

**Caracteristiques** :
- Chaque classe a le meme poids, peu importe sa frequence
- Ideal quand toutes les classes sont aussi importantes
- Dans notre projet : les departs et arrivees sont egalement critiques

### F1 Micro

Agrege tous les TP, FP, FN avant de calculer :

```
TP_total = TP_DEPART + TP_ARRIVEE
FP_total = FP_DEPART + FP_ARRIVEE
FN_total = FN_DEPART + FN_ARRIVEE

Precision_micro = TP_total / (TP_total + FP_total)
Rappel_micro = TP_total / (TP_total + FN_total)
F1_micro = 2 × Precision_micro × Rappel_micro / (Precision_micro + Rappel_micro)
```

**Caracteristiques** :
- Les classes frequentes dominent le score
- Ideal quand on a des classes tres desequilibrees et qu'on veut privilegier les classes majoritaires

### Choix du projet

Le projet utilise **F1 Macro** car :
1. DEPART et ARRIVEE sont egalement importants pour generer un itineraire
2. Les deux classes sont presentes dans des proportions similaires

---

## 6. Exact Match

**Question** : *Quelle proportion des phrases ont DEPART ET ARRIVEE tous deux corrects ?*

```
                        Nb phrases avec (DEP correct ET ARR correct)
Exact Match = ──────────────────────────────────────────────────────────
                            Nb total de phrases valides
```

### Difference avec F1

Le F1 evalue chaque entite independamment. L'Exact Match est plus strict : une phrase n'est comptee comme succes que si les DEUX entites sont correctes.

### Exemple

Sur 100 phrases :
- 85 ont le depart ET l'arrivee corrects
- 10 ont seulement le depart correct
- 5 ont seulement l'arrivee correcte

**Exact Match = 85%** (seules les 85 phrases parfaites comptent)

### Importance pour le projet

C'est la metrique la plus representative de l'usage reel : une phrase avec un depart correct mais une arrivee fausse produira un itineraire faux. L'Exact Match mesure donc le taux de succes reel du pipeline.

| Modele | F1 DEPART | F1 ARRIVEE | Exact Match |
|--------|-----------|------------|-------------|
| Baseline | 73.14% | 71.12% | 63.48% |
| spaCy custom | 95.72% | 94.82% | 85.22% |
| CamemBERT | 99.82% | 99.52% | ~99% |

---

## 7. Implementation dans le projet

La fonction `compute_metrics()` dans `src/nlp/camembert_ner.py:158-206` implemente ces calculs :

```python
def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute NER metrics (precision, recall, F1) for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    metrics = {}
    for entity in ["DEPART", "ARRIVEE"]:
        b_label = LABEL2ID[f"B-{entity}"]
        i_label = LABEL2ID[f"I-{entity}"]

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for pred_seq, label_seq in zip(predictions, labels):
            # Extraction des spans d'entites predits et reels
            pred_entities = _extract_entities(pred_seq, b_label, i_label)
            true_entities = _extract_entities(label_seq, b_label, i_label)

            # Comptage des TP, FP, FN
            for pe in pred_entities:
                if pe in true_entities:
                    true_positives += 1
                else:
                    false_positives += 1

            for te in true_entities:
                if te not in pred_entities:
                    false_negatives += 1

        # Calcul des metriques
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

        metrics[f"precision_{entity.lower()}"] = precision
        metrics[f"recall_{entity.lower()}"] = recall
        metrics[f"f1_{entity.lower()}"] = f1

    # F1 Macro = moyenne des F1 par entite
    metrics["f1_macro"] = (metrics["f1_depart"] + metrics["f1_arrivee"]) / 2

    return metrics
```

### Points cles de l'implementation

1. **Schema BIO** : Les entites utilisent le format B-DEPART (debut) et I-DEPART (continuation)
2. **Comparaison de spans** : Deux entites sont identiques si leurs positions (start, end) correspondent exactement
3. **Evaluation par entite** : Chaque type (DEPART, ARRIVEE) est evalue separement
4. **Division par zero** : Le code gere les cas ou denominateur = 0 (retourne 0)

---

## 8. Resume visuel

```
┌─────────────────────────────────────────────────────────────────────┐
│                      METRIQUES NER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PRECISION          RAPPEL              F1-SCORE                   │
│   ──────────         ──────              ────────                   │
│      TP                 TP             2 × P × R                    │
│   ────────           ────────          ─────────                    │
│   TP + FP            TP + FN            P + R                       │
│                                                                     │
│   "Suis-je fiable    "Ai-je tout       "Equilibre entre            │
│    quand je          trouve ?"          precision et                │
│    predis ?"                            rappel"                     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   F1 MACRO                          EXACT MATCH                     │
│   ────────                          ───────────                     │
│   (F1_DEP + F1_ARR) / 2            % phrases parfaites              │
│                                                                     │
│   "Performance moyenne              "Succes reel du                 │
│    par classe"                       pipeline complet"              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

★ Insight ─────────────────────────────────────

**Pourquoi le F1 et pas l'accuracy ?** En NER, la grande majorite des tokens sont "O" (hors entite). Un modele qui predit toujours "O" aurait une accuracy de ~95% mais serait completement inutile. Le F1, en ignorant les TN, se concentre sur ce qui compte : les entites.

**Evaluation par span, pas par token** : La fonction `_extract_entities()` regroupe les tokens B- et I- en spans complets. Cela signifie que "Saint-Pierre-des-Corps" doit etre entierement correct : si le modele predit seulement "Saint-Pierre", c'est compte comme FP + FN, pas comme succes partiel.

**Le F1 Macro comme critere d'early stopping** : Le Trainer HuggingFace utilise `metric_for_best_model="f1_macro"` pour sauvegarder le meilleur modele. Cela garantit un modele equilibre sur les deux classes.

─────────────────────────────────────────────────
