# Early Stopping : Arrêt Précoce de l'Entraînement

L'early stopping est une technique de régularisation qui arrête l'entraînement lorsque le modèle commence à sur-apprendre les données d'entraînement.

---

## 1. Le Problème du Sur-apprentissage (Overfitting)

Lors de l'entraînement d'un modèle de deep learning, on observe typiquement trois phases :

```
Performance
    │
    │     ┌─────────────────── Eval (validation)
    │    /    \
    │   /      \_____________ Début overfitting
    │  /
    │ /  _____________________ Train
    │/  /
    └──────────────────────────────────────────► Steps
       │         │           │
    Phase 1   Phase 2     Phase 3
   (apprentissage) (optimal)  (overfitting)
```

| Phase | Train Loss | Eval Loss | Description |
|-------|-----------|-----------|-------------|
| **Phase 1** | ↓ Diminue | ↓ Diminue | Le modèle apprend des patterns utiles |
| **Phase 2** | ↓ Diminue | → Stagne | Point optimal - arrêter ici ! |
| **Phase 3** | ↓ Diminue | ↑ Augmente | Le modèle mémorise les données d'entraînement |

### Signes de sur-apprentissage

- La loss d'entraînement continue de baisser mais la loss de validation remonte
- Le F1-score sur le jeu de test plafonne ou diminue
- Le modèle prédit parfaitement les exemples d'entraînement mais échoue sur de nouveaux exemples

---

## 2. Comment Détecter le Moment Optimal

L'idée est de surveiller une métrique sur le jeu de **validation** (pas d'entraînement !) et d'arrêter quand elle ne s'améliore plus.

### Métrique surveillée dans notre projet : F1 Macro

```python
# Dans TrainingArguments (camembert_ner.py:313-314)
metric_for_best_model="f1_macro",
greater_is_better=True,
```

Le F1 macro est la moyenne des F1-scores de chaque classe :

```
F1_macro = (F1_DEPART + F1_ARRIVEE) / 2
```

Calculé dans `compute_metrics()` (lignes 158-206) :

```python
metrics["f1_macro"] = (metrics["f1_depart"] + metrics["f1_arrivee"]) / 2
```

### Pourquoi F1 plutôt que la Loss ?

- La loss peut fluctuer même quand les prédictions s'améliorent
- Le F1 mesure directement ce qui nous intéresse : la qualité des extractions
- Pour le NER, un token mal classé peut avoir peu d'impact sur la loss mais ruiner le F1

---

## 3. La Patience : Tolérance aux Fluctuations

La **patience** définit combien d'évaluations consécutives sans amélioration sont tolérées avant d'arrêter.

### Configuration dans notre projet

```python
# Dans TrainingConfig (camembert_ner.py:69)
early_stopping_patience: int = 3
```

Cela signifie :
- Évaluation toutes les 100 steps (`eval_steps = 100`)
- Si le F1 macro ne s'améliore pas pendant **3 évaluations consécutives** (300 steps)
- L'entraînement s'arrête automatiquement

### Visualisation de la patience

```
F1_macro
  0.98 │                    ╭──────────── Best model sauvegardé
       │                   /
  0.96 │              ╭───╯
       │             /
  0.94 │        ╭───╯
       │       /     \___/\___/\___  ← 3 évals sans amélioration
  0.92 │  ╭───╯                    │
       │ /                         STOP!
  0.90 │/
       └──────────────────────────────────────► Steps
           100   200   300   400   500   600
                                   │
                              Patience=3 atteinte
```

### Choix de la patience

| Patience | Comportement | Quand l'utiliser |
|----------|-------------|------------------|
| 1-2 | Très agressif | Dataset stable, peu de bruit |
| **3** | Équilibré | **Notre choix** - bon compromis |
| 5-10 | Conservateur | Dataset bruité, métriques instables |

---

## 4. Best Model vs Last Model

### Le problème sans sauvegarde du meilleur modèle

```
F1 sur validation
       │
  0.98 │         ★ Best model (step 400)
       │        /\
  0.96 │       /  \
       │      /    \
  0.94 │     /      \____
       │    /            \
  0.92 │   /              \_____ ← Last model (step 700)
       │  /                     │
       └────────────────────────┴──► Steps
               400        700
```

Si on garde uniquement le dernier modèle, on perd la meilleure version !

### La solution : `load_best_model_at_end`

```python
# Dans TrainingArguments (camembert_ner.py:312)
load_best_model_at_end=True,
```

Cette option :
1. Sauvegarde un checkpoint à chaque amélioration du F1 macro
2. À la fin de l'entraînement, recharge automatiquement le meilleur checkpoint
3. Le modèle final exporté est donc le **meilleur**, pas le dernier

### Limitation du nombre de checkpoints

```python
# Dans TrainingArguments (camembert_ner.py:317)
save_total_limit=2,
```

Pour ne pas remplir le disque, on garde uniquement les 2 derniers checkpoints.

---

## 5. Implémentation avec HuggingFace

### L'EarlyStoppingCallback

```python
from transformers import EarlyStoppingCallback

# Dans CamemBERTTrainer.train() (lignes 329-331)
trainer = Trainer(
    # ... autres arguments ...
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)
    ],
)
```

### Fonctionnement interne du callback

```python
# Pseudo-code simplifié du fonctionnement interne
class EarlyStoppingCallback:
    def __init__(self, patience):
        self.patience = patience
        self.patience_counter = 0
        self.best_metric = None

    def on_evaluate(self, state, metrics):
        current_metric = metrics["eval_f1_macro"]

        if self.best_metric is None or current_metric > self.best_metric:
            # Amélioration ! Reset du compteur
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            # Pas d'amélioration
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            # Arrêt de l'entraînement
            state.should_training_stop = True
```

### Configuration complète requise

Pour que l'early stopping fonctionne correctement, plusieurs paramètres doivent être cohérents :

```python
training_args = TrainingArguments(
    # 1. Stratégie d'évaluation
    eval_strategy="steps",        # Évaluer régulièrement
    eval_steps=100,               # Toutes les 100 steps

    # 2. Stratégie de sauvegarde (doit correspondre à eval)
    save_strategy="steps",        # Sauvegarder régulièrement
    save_steps=100,               # Aux mêmes steps que l'évaluation

    # 3. Métrique à surveiller
    metric_for_best_model="f1_macro",  # Quelle métrique optimiser
    greater_is_better=True,            # Plus grand = meilleur

    # 4. Sauvegarde du meilleur
    load_best_model_at_end=True,  # Recharger le meilleur à la fin
    save_total_limit=2,           # Garder max 2 checkpoints
)
```

---

## 6. Exemple Concret d'Entraînement

Voici un log typique montrant l'early stopping en action :

```
Step 100 | Train Loss: 0.45 | Eval F1: 0.82 | ★ New best!
Step 200 | Train Loss: 0.32 | Eval F1: 0.91 | ★ New best!
Step 300 | Train Loss: 0.21 | Eval F1: 0.95 | ★ New best!
Step 400 | Train Loss: 0.15 | Eval F1: 0.97 | ★ New best!
Step 500 | Train Loss: 0.11 | Eval F1: 0.96 | Patience: 1/3
Step 600 | Train Loss: 0.08 | Eval F1: 0.96 | Patience: 2/3
Step 700 | Train Loss: 0.05 | Eval F1: 0.95 | Patience: 3/3 → STOP

Training stopped early at step 700
Best model from step 400 loaded (F1: 0.97)
```

Dans cet exemple :
- Le meilleur F1 (0.97) est atteint au step 400
- Les 3 évaluations suivantes (500, 600, 700) n'améliorent pas
- L'entraînement s'arrête au step 700
- Le modèle du step 400 est automatiquement rechargé

---

## 7. Avantages de l'Early Stopping

| Avantage | Explication |
|----------|-------------|
| **Prévient l'overfitting** | Arrête avant que le modèle ne mémorise |
| **Économise du temps** | Pas besoin de terminer tous les epochs |
| **Économise des ressources** | Moins de GPU/CPU utilisés |
| **Automatique** | Pas besoin de surveiller manuellement |
| **Meilleure généralisation** | Le modèle final performe mieux sur de nouvelles données |

---

## 8. Résumé de Notre Configuration

```python
@dataclass
class TrainingConfig:
    # ...
    eval_steps: int = 100                    # Évaluer toutes les 100 steps
    early_stopping_patience: int = 3         # 3 évals sans amélioration = stop
    # ...

# Métrique surveillée : F1 macro (moyenne F1_DEPART et F1_ARRIVEE)
# Sauvegarde : meilleur modèle automatiquement rechargé à la fin
```

Cette configuration permet d'obtenir un modèle avec un **F1 macro de 99.67%** tout en évitant le sur-apprentissage, comme documenté dans les métriques du projet.

---

★ Insight ─────────────────────────────────────
**Pourquoi patience=3 ?** : Avec `eval_steps=100`, une patience de 3 signifie 300 steps de tolérance. C'est suffisant pour absorber les fluctuations normales du F1 sans attendre trop longtemps après le pic optimal.

**Trade-off temps/performance** : Une patience plus élevée pourrait trouver un meilleur modèle dans certains cas, mais le gain marginal ne justifie généralement pas le temps de calcul supplémentaire, surtout avec un F1 déjà à 99%+.
─────────────────────────────────────────────────
