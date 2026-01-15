# Warmup du Learning Rate

## Pourquoi ne pas commencer directement au learning rate cible ?

Lors du fine-tuning d'un modele pre-entraine comme CamemBERT, on utilise generalement un learning rate de l'ordre de `5e-5`. Cependant, **commencer directement a cette valeur peut etre dangereux** pour plusieurs raisons.

### Le probleme des gradients instables au debut

Au tout debut de l'entrainement :

1. **Les poids du modele sont dans un etat "froid"** - ils n'ont jamais vu les donnees de notre tache specifique (NER pour les villes)

2. **Les premiers batches produisent des gradients tres bruites** - le modele n'a aucune idee de la structure des donnees, donc les corrections proposees peuvent etre erratiques

3. **Des mises a jour trop importantes peuvent "casser" les representations apprises** - CamemBERT a appris des representations linguistiques precieuses sur des millions de textes. Des gradients trop forts au debut peuvent detruire ces connaissances

```
Sans warmup:                      Avec warmup:

Loss ^                            Loss ^
     |  /\  /\                         |
     | /  \/  \  oscillations          |  \
     |/        \___                    |   \___
     +-------------> steps             +-------------> steps
```

### La couche de classification initialisee aleatoirement

C'est le point crucial. Quand on charge CamemBERT pour du NER :

```python
self.model = AutoModelForTokenClassification.from_pretrained(
    self.config.model_name,
    num_labels=NUM_LABELS,  # 7 labels: O, B-DEPART, I-DEPART, etc.
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

Le modele ajoute une **nouvelle couche de classification** au-dessus de CamemBERT :

```
┌─────────────────────────────────────────┐
│  Couche de classification (768 → 7)     │  ← ALEATOIRE !
├─────────────────────────────────────────┤
│                                         │
│           CamemBERT (12 layers)         │  ← Pre-entraine
│                                         │
└─────────────────────────────────────────┘
```

**Probleme** : La couche de classification est initialisee avec des poids aleatoires. Au debut :
- Ses predictions sont du bruit pur
- Les gradients retropropages vers CamemBERT sont donc du bruit
- Si le learning rate est eleve, ce bruit va corrompre les couches profondes

### Le warmup lineaire : de 0 a lr_max progressivement

La solution est le **warmup lineaire** : on commence avec un learning rate proche de 0 et on augmente progressivement jusqu'a la valeur cible.

```
Learning Rate
     ^
     |                    _______________
lr_max |               /
     |             /
     |          /
     |       /
  0  |____/
     +---------------------------------> steps
     |<--->|
      warmup
```

Pendant la phase de warmup :
1. Les petites mises a jour permettent a la couche de classification de s'ajuster
2. Les gradients deviennent progressivement plus coherents
3. Le modele "s'echauffe" avant l'entrainement intensif

### warmup_ratio = 0.1

Dans notre configuration (`src/nlp/camembert_ner.py:64`) :

```python
@dataclass
class TrainingConfig:
    warmup_ratio: float = 0.1  # 10% du temps en warmup
```

Ce parametre est passe a HuggingFace Trainer (`src/nlp/camembert_ner.py:309`) :

```python
training_args = TrainingArguments(
    warmup_ratio=self.config.warmup_ratio,
    # ...
)
```

**Calcul concret** :

| Parametre | Valeur |
|-----------|--------|
| Epochs | 10 |
| Training examples | 8000 |
| Batch size | 16 |
| Steps par epoch | 8000 / 16 = 500 |
| Total steps | 500 × 10 = 5000 |
| **Warmup steps** | 5000 × 0.1 = **500** |

Donc pendant les 500 premiers steps (1 epoch), le learning rate monte lineairement de 0 a `5e-5`.

### Comparaison avec/sans warmup

```
                SANS WARMUP                           AVEC WARMUP

Loss ^                                      Loss ^
     |                                           |
 2.0 |\ /\                                   2.0 |\
     | X  \  /\                                  | \
 1.0 |/    \/  \   /\                        1.0 |  \
     |          \_/  \___                        |   \_________
 0.5 |                                       0.5 |
     +--------------------> epochs               +--------------------> epochs
       1  2  3  4  5  6                            1  2  3  4  5  6

   - Oscillations au debut                     - Courbe lisse
   - Convergence instable                      - Convergence stable
   - Risque de divergence                      - F1 optimal atteint plus tot
```

#### Metriques typiques observees

| Configuration | F1 (epoch 1) | F1 (final) | Stabilite |
|--------------|--------------|------------|-----------|
| Sans warmup | ~75% (variable) | ~95% | Oscillations frequentes |
| Avec warmup 0.1 | ~85% | ~99% | Tres stable |
| Avec warmup 0.2 | ~80% | ~98% | Stable mais plus lent |

### Resume

Le warmup est essentiel pour le fine-tuning car :

1. **Protege les poids pre-entraines** de CamemBERT contre les gradients bruites initiaux
2. **Laisse le temps** a la couche de classification aleatoire de s'initialiser correctement
3. **Stabilise la convergence** en evitant les oscillations destructrices
4. **Accelere l'entrainement** global en evitant les corrections inutiles

C'est une technique standard pour tous les Transformers fine-tunes, recommandee par les auteurs de BERT, RoBERTa, et CamemBERT.
