⏺ Fonctionnement de l'entraînement CamemBERT

L'entraînement CamemBERT pour la reconnaissance d'entités nommées (NER) utilise le fine-tuning d'un modèle Transformer pré-entraîné. Voici une explication détaillée du processus.

---
1. Architecture Générale

CamemBERT est un modèle de type BERT entraîné spécifiquement sur du texte français (138 Go de données). Le fine-tuning ajoute une couche de classification au-dessus pour prédire les labels NER.

```
┌─────────────────────────────────────────────────────────────┐
│                    Architecture du modèle                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Entrée: "Je veux aller de Paris à Lyon"                   │
│              ↓                                               │
│   ┌────────────────────────────────────┐                    │
│   │         Tokenizer CamemBERT         │                    │
│   │   (découpe en sous-mots BPE)        │                    │
│   └────────────────────────────────────┘                    │
│              ↓                                               │
│   [<s>, Je, veux, aller, de, Paris, à, Lyon, </s>]          │
│              ↓                                               │
│   ┌────────────────────────────────────┐                    │
│   │      CamemBERT (12 couches)         │                    │
│   │   Transformer Bidirectionnel        │                    │
│   └────────────────────────────────────┘                    │
│              ↓                                               │
│   ┌────────────────────────────────────┐                    │
│   │   Couche de Classification (7 labels)│                   │
│   │   O, B-DEPART, I-DEPART,            │                    │
│   │   B-ARRIVEE, I-ARRIVEE,             │                    │
│   │   B-VIA, I-VIA                      │                    │
│   └────────────────────────────────────┘                    │
│              ↓                                               │
│   Sortie: [O, O, O, O, O, B-DEPART, O, B-ARRIVEE, O]        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---
2. Le Schéma BIO (Begin-Inside-Outside)

Le NER utilise un schéma d'annotation appelé BIO pour gérer les entités multi-mots :

┌─────────┬────────────────────────────────────────────────────┐
│  Label  │                   Signification                     │
├─────────┼────────────────────────────────────────────────────┤
│ O       │ Token hors entité (Outside)                         │
├─────────┼────────────────────────────────────────────────────┤
│ B-DEPART│ Premier token d'une entité DÉPART (Begin)           │
├─────────┼────────────────────────────────────────────────────┤
│ I-DEPART│ Token suivant dans une entité DÉPART (Inside)       │
├─────────┼────────────────────────────────────────────────────┤
│ B-ARRIVEE│ Premier token d'une entité ARRIVÉE                 │
├─────────┼────────────────────────────────────────────────────┤
│ I-ARRIVEE│ Token suivant dans une entité ARRIVÉE              │
├─────────┼────────────────────────────────────────────────────┤
│ B-VIA   │ Premier token d'un point de passage (VIA)           │
├─────────┼────────────────────────────────────────────────────┤
│ I-VIA   │ Token suivant dans une entité VIA                   │
└─────────┴────────────────────────────────────────────────────┘

Exemple avec une ville composée :

```
Phrase: "Je pars de Saint-Pierre-des-Corps"

Tokens:  Je   pars  de  Saint  -  Pierre  -  des  -  Corps
Labels:  O    O     O   B-DEP  I  I-DEP   I  I    I  I-DEP
```

---
3. Configuration d'entraînement (TrainingConfig)

La classe `TrainingConfig` définit tous les hyperparamètres :

```python
@dataclass
class TrainingConfig:
    model_name: str = "camembert-base"    # Modèle de base HuggingFace
    max_length: int = 128                  # Longueur max des séquences
    batch_size: int = 16                   # Phrases par batch
    learning_rate: float = 5e-5            # Taux d'apprentissage
    num_epochs: int = 10                   # Nombre d'époques
    warmup_ratio: float = 0.1              # 10% du temps en warmup
    weight_decay: float = 0.01             # Régularisation L2
    eval_steps: int = 100                  # Évaluation toutes les 100 steps
    early_stopping_patience: int = 3       # Arrêt si pas d'amélioration
    fp16: bool = True                      # Précision mixte (GPU)
```

Explication des paramètres clés :

┌────────────────────┬──────────────────────────────────────────────────┐
│     Paramètre      │                   Explication                     │
├────────────────────┼──────────────────────────────────────────────────┤
│ learning_rate=5e-5 │ Taux très faible car le modèle est déjà          │
│                    │ pré-entraîné. Trop élevé = perte des             │
│                    │ connaissances acquises (catastrophic forgetting) │
├────────────────────┼──────────────────────────────────────────────────┤
│ warmup_ratio=0.1   │ Pendant 10% de l'entraînement, le learning rate  │
│                    │ augmente progressivement de 0 à 5e-5.            │
│                    │ Évite les gradients instables au début.          │
├────────────────────┼──────────────────────────────────────────────────┤
│ weight_decay=0.01  │ Pénalise les poids trop grands pour éviter       │
│                    │ le sur-apprentissage (régularisation L2)         │
├────────────────────┼──────────────────────────────────────────────────┤
│ early_stopping=3   │ Arrête l'entraînement si le F1 n'améliore pas    │
│                    │ pendant 3 évaluations consécutives               │
└────────────────────┴──────────────────────────────────────────────────┘

---
4. Le Dataset NER (NERDataset)

La classe `NERDataset` charge les données depuis le format spaCy `.spacy` et les convertit pour PyTorch :

```
Processus de chargement :

1. Charger le fichier .spacy avec spaCy
   └── DocBin contient les phrases annotées

2. Pour chaque document :
   └── Extraire le texte et les entités [(start, end, label), ...]

3. Tokenization avec CamemBERT :
   └── "Paris" → ["▁Paris"] (un seul token)
   └── "Saint-Étienne" → ["▁Saint", "-", "É", "tienne"] (4 tokens)

4. Alignement des labels avec les tokens :
   └── Utilise offset_mapping pour aligner caractères → tokens
```

L'alignement est crucial car CamemBERT utilise la tokenization BPE qui découpe les mots :

```
Texte:    "Clermont-Ferrand"
          0       8       16

Tokens:   [▁Cler, mont, -, Fer, rand]
Offsets:  [(0,4), (4,8), (8,9), (9,12), (12,16)]

Labels:   [B-DEPART, I-DEPART, I-DEPART, I-DEPART, I-DEPART]
```

---
5. La Fonction de Métriques (compute_metrics)

Cette fonction calcule Précision, Rappel et F1 pour chaque type d'entité :

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)  # Prédiction la plus probable

    # Pour chaque entité (DEPART, ARRIVEE) :
    # 1. Extraire les spans prédits
    # 2. Extraire les spans réels
    # 3. Calculer TP, FP, FN
```

Calcul des métriques :

```
                      Prédictions
                   DEPART    Autre
              ┌─────────┬─────────┐
Réalité DEPART│   TP    │   FN    │
              ├─────────┼─────────┤
        Autre │   FP    │   TN    │
              └─────────┴─────────┘

Précision = TP / (TP + FP)   → "Parmi mes prédictions DEPART, combien sont correctes ?"
Rappel    = TP / (TP + FN)   → "Parmi les vrais DEPART, combien ai-je trouvés ?"
F1        = 2 * (P * R) / (P + R)  → Moyenne harmonique
```

---
6. Le Processus d'Entraînement (CamemBERTTrainer)

La méthode `train()` orchestre tout le processus :

```
┌─────────────────────────────────────────────────────────────┐
│                 Pipeline d'entraînement                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Charger le tokenizer CamemBERT                          │
│     └── AutoTokenizer.from_pretrained("camembert-base")     │
│                                                              │
│  2. Charger le modèle avec couche de classification         │
│     └── AutoModelForTokenClassification(num_labels=7)       │
│                                                              │
│  3. Créer les datasets train/eval                           │
│     └── NERDataset(train.spacy), NERDataset(test.spacy)     │
│                                                              │
│  4. Configurer le Trainer HuggingFace                       │
│     └── TrainingArguments + EarlyStoppingCallback           │
│                                                              │
│  5. Lancer l'entraînement                                   │
│     └── trainer.train()                                     │
│                                                              │
│  6. Sauvegarder le meilleur modèle                          │
│     └── trainer.save_model()                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---
7. L'Inférence (CamemBERTNER)

La classe `CamemBERTNER` utilise le modèle entraîné pour extraire les entités :

```python
def extract(self, sentence: str) -> NERResult:
    # 1. Tokenizer la phrase
    encoding = self.tokenizer(sentence, return_offsets_mapping=True)

    # 2. Passer dans le modèle
    with torch.no_grad():
        outputs = self.model(input_ids, attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)

    # 3. Reconstruire les entités depuis les tokens
    # 4. Valider contre la liste des villes connues
    # 5. Retourner NERResult(departure, arrival, vias, is_valid)
```

Post-traitements importants :
- Validation des villes : vérifie que l'entité extraite est une ville connue
- Correction des fautes : utilise rapidfuzz pour corriger "Marseile" → "Marseille"
- Détection des VIA : analyse le contexte ("en passant par", "via") pour identifier les étapes
- Nettoyage : supprime les prépositions parasites ("de Paris" → "Paris")

---
8. Comparaison avec spaCy

┌────────────────────┬─────────────────────┬─────────────────────┐
│     Aspect         │       spaCy         │     CamemBERT       │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Architecture       │ CNN (Tok2Vec)       │ Transformer (12L)   │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Contexte           │ Local (±4 tokens)   │ Global (128 tokens) │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Pré-entraînement   │ Non                 │ Oui (138 Go fr)     │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Taille modèle      │ ~10 Mo              │ ~440 Mo             │
├────────────────────┼─────────────────────┼─────────────────────┤
│ Vitesse inférence  │ ~50,000 tokens/sec  │ ~5,000 tokens/sec   │
├────────────────────┼─────────────────────┼─────────────────────┤
│ F1 Score           │ ~95.5%              │ ~99.7%              │
├────────────────────┼─────────────────────┼─────────────────────┤
│ GPU requis         │ Non                 │ Recommandé          │
└────────────────────┴─────────────────────┴─────────────────────┘

---
★ Insight ─────────────────────────────────────
**Pourquoi le fine-tuning est efficace** : CamemBERT a déjà appris la grammaire et le vocabulaire français. Le fine-tuning n'a qu'à lui apprendre la tâche spécifique (reconnaître DEPART/ARRIVEE), ce qui nécessite peu de données (~8000 exemples) et peu d'époques (~5).

**Le warmup est crucial** : Sans warmup, les premiers gradients seraient calculés sur un modèle "froid" avec une couche de classification aléatoire, causant des mises à jour destructrices. Le warmup permet une adaptation progressive.

**Attention au sur-apprentissage** : Avec seulement 7 labels et 8000 exemples, le modèle peut facilement mémoriser les données. L'early stopping et le weight decay sont essentiels pour généraliser.

**Tokenization BPE et noms propres** : Les noms de villes rares comme "Pontarlier" sont découpés en sous-mots. Le modèle doit apprendre à reconstituer l'entité à partir de plusieurs tokens, d'où l'importance du schéma BIO avec les labels I-*.
─────────────────────────────────────────────────
