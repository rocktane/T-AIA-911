# Fine-tuning : Adapter un modele pre-entraine a une tache specifique

## Introduction

Le **fine-tuning** est une technique d'apprentissage automatique qui permet d'adapter un modele pre-entraine sur une grande quantite de donnees generales a une tache specifique avec relativement peu de donnees. Dans notre projet, nous utilisons cette technique pour adapter CamemBERT (un modele de langage francais) a la reconnaissance d'entites nommees (NER) pour extraire les villes de depart, d'arrivee et les etapes intermediaires.

---

## 1. Le Transfer Learning : Pourquoi repartir de zero est inefficace

### Le probleme de l'apprentissage classique

Entrainer un modele de deep learning a partir de zero necessite :
- **Des millions de donnees** annotees
- **Des semaines de calcul** sur GPU puissants
- **Une expertise considerable** en optimisation

Pour notre tache de NER sur 10,000 phrases, entrainer un transformer de zero serait :
- Impossible (pas assez de donnees)
- Couteux (ressources de calcul)
- Lent (temps d'entrainement)

### La solution : Transfer Learning

Le **Transfer Learning** (apprentissage par transfert) permet de reutiliser les connaissances acquises par un modele sur une tache source pour resoudre une tache cible differente.

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSFER LEARNING                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Tache Source                    Tache Cible                │
│  (Pre-entrainement)              (Fine-tuning)              │
│                                                             │
│  ┌─────────────────┐            ┌─────────────────┐         │
│  │ Corpus massif   │            │ Dataset NER     │         │
│  │ Wikipedia FR    │ ────────►  │ 10,000 phrases  │         │
│  │ Livres, Web     │  Transfer  │ annotees        │         │
│  │ ~138 Go texte   │            │                 │         │
│  └─────────────────┘            └─────────────────┘         │
│                                                             │
│  Apprentissage:                 Apprentissage:              │
│  - Structure du francais        - Reconnaissance villes     │
│  - Grammaire, syntaxe           - Patterns "de X a Y"       │
│  - Semantique des mots          - Classification B-I-O      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Analogie** : C'est comme apprendre une nouvelle langue. Quelqu'un qui connait deja plusieurs langues apprendra plus vite qu'un debutant car il comprend deja les concepts de grammaire, conjugaison, etc.

---

## 2. Pre-entrainement vs Fine-tuning

### Phase 1 : Pre-entrainement (realise par les chercheurs)

CamemBERT a ete pre-entraine par l'equipe de recherche sur un corpus massif :

| Aspect | Details |
|--------|---------|
| **Corpus** | OSCAR (138 Go de texte francais) |
| **Tache** | Masked Language Modeling (MLM) |
| **Duree** | ~100 heures sur 256 GPUs |
| **Resultat** | Comprehension profonde du francais |

**Tache MLM** : Le modele apprend a predire des mots masques dans des phrases.

```
Entree:  "Je veux [MASK] de Paris a [MASK]"
Sortie:  "Je veux partir de Paris a Lyon"
```

Cette tache force le modele a comprendre :
- Le **contexte** (quel mot a du sens ici ?)
- La **syntaxe** (quelle forme grammaticale ?)
- La **semantique** (quelle signification ?)

### Phase 2 : Fine-tuning (ce que nous faisons)

Nous adaptons CamemBERT a notre tache specifique de NER :

```python
# Dans src/nlp/camembert_ner.py - CamemBERTTrainer.train()

# Chargement du modele pre-entraine avec une nouvelle tete de classification
self.model = AutoModelForTokenClassification.from_pretrained(
    self.config.model_name,  # "camembert-base"
    num_labels=NUM_LABELS,   # 7 labels (O, B-DEPART, I-DEPART, ...)
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
```

| Aspect | Details |
|--------|---------|
| **Dataset** | 10,000 phrases (8,000 train / 2,000 test) |
| **Tache** | Token Classification (NER) |
| **Duree** | ~30 minutes sur GPU, ~2h sur CPU |
| **Resultat** | NER specialise pour les trajets |

---

## 3. Architecture : Conserver les poids, ajouter une couche

### Ce qui se passe lors du chargement

Quand on appelle `AutoModelForTokenClassification.from_pretrained()` :

```
┌─────────────────────────────────────────────────────────────┐
│                    ARCHITECTURE DU MODELE                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           COUCHE DE CLASSIFICATION (nouvelle)       │    │
│  │                                                     │    │
│  │    Linear(768 → 7)  ← Poids ALEATOIRES             │    │
│  │    [O, B-DEP, I-DEP, B-ARR, I-ARR, B-VIA, I-VIA]   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              CAMEMBERT (pre-entraine)               │    │
│  │                                                     │    │
│  │    12 couches Transformer                          │    │
│  │    Poids PRE-ENTRAINES conserves                   │    │
│  │    ~110 millions de parametres                     │    │
│  │                                                     │    │
│  │    Comprend deja:                                  │    │
│  │    - Syntaxe francaise                             │    │
│  │    - Noms propres (villes, personnes)              │    │
│  │    - Patterns linguistiques                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ▲                                  │
│                          │                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           TOKENIZER + EMBEDDINGS                    │    │
│  │                                                     │    │
│  │    Vocabulaire: 32,005 tokens                      │    │
│  │    Embeddings: 768 dimensions                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Pourquoi ajouter une couche ?

Le modele pre-entraine CamemBERT produit des **representations vectorielles** (embeddings) de 768 dimensions pour chaque token. Ces vecteurs capturent la semantique, mais ne sont pas directement utilisables pour la classification.

La nouvelle couche lineaire transforme ces representations en **scores de classification** :

```
Token "Paris" → Embedding [0.23, -0.45, ..., 0.12] (768 dim)
                            ↓
              Linear(768 → 7)
                            ↓
              Scores [0.1, 8.5, 0.2, 0.3, 0.1, 0.1, 0.1]
                            ↓
              Softmax → Probabilites
                            ↓
              Prediction: B-DEPART (score 8.5 → 99.8%)
```

### Les 7 labels de notre tache

| Label | Signification | Exemple |
|-------|--------------|---------|
| O | Outside (hors entite) | "Je", "veux", "aller" |
| B-DEPART | Begin Departure | "**Paris**" dans "de Paris" |
| I-DEPART | Inside Departure | "**Etienne**" dans "Saint-Etienne" |
| B-ARRIVEE | Begin Arrival | "**Lyon**" dans "a Lyon" |
| I-ARRIVEE | Inside Arrival | "**Dieu**" dans "Part-Dieu" |
| B-VIA | Begin Via | "**Dijon**" dans "via Dijon" |
| I-VIA | Inside Via | "**Ferrand**" dans "Clermont-Ferrand" |

---

## 4. Le Catastrophic Forgetting : Le danger principal

### Definition

Le **Catastrophic Forgetting** (oubli catastrophique) est un phenomene ou un reseau de neurones "oublie" brutalement les connaissances apprises precedemment lorsqu'il apprend une nouvelle tache.

```
┌─────────────────────────────────────────────────────────────┐
│               CATASTROPHIC FORGETTING                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AVANT fine-tuning:          APRES fine-tuning agressif:   │
│                                                             │
│  Le modele sait que:         Le modele a oublie:           │
│  ✓ "Paris" est une ville     ✗ Structure syntaxique        │
│  ✓ "de X a Y" = trajet       ✗ Noms propres rares          │
│  ✓ Grammaire francaise       ✗ Comprehension contextuelle  │
│                                                             │
│  Cause: Learning rate trop eleve                           │
│         → Les poids pre-entraines sont ecrases             │
│         → Le modele perd ses connaissances generales       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Analogie

Imaginez un chef cuisinier experimente (modele pre-entraine) qui connait toutes les techniques de cuisine. Si on lui demande d'apprendre a faire des sushis (nouvelle tache) de maniere trop intensive, il risque d'oublier comment faire une sauce bearnaise !

### Comment l'eviter dans notre projet

Notre implementation utilise plusieurs strategies pour prevenir le catastrophic forgetting :

```python
# Dans src/nlp/camembert_ner.py - TrainingConfig

@dataclass
class TrainingConfig:
    learning_rate: float = 5e-5      # ① Tres faible LR
    warmup_ratio: float = 0.1        # ② Warmup progressif
    weight_decay: float = 0.01       # ③ Regularisation
    early_stopping_patience: int = 3 # ④ Arret precoce
```

---

## 5. Learning Rate faible : La cle du succes

### Pourquoi 5e-5 ?

Le **learning rate** (taux d'apprentissage) controle l'amplitude des modifications des poids a chaque iteration.

| Learning Rate | Effet sur les poids pre-entraines |
|---------------|----------------------------------|
| 1e-3 (standard) | Modifications MASSIVES → Oubli catastrophique |
| 1e-4 (moyen) | Modifications IMPORTANTES → Risque d'oubli |
| **5e-5 (notre choix)** | Modifications FINES → Preservation des connaissances |
| 1e-6 (tres faible) | Modifications MINIMES → Apprentissage trop lent |

### Intuition visuelle

```
┌─────────────────────────────────────────────────────────────┐
│            IMPACT DU LEARNING RATE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Paysage de la fonction de perte (Loss Landscape)          │
│                                                             │
│       ▲ Loss                                                │
│       │      ╱╲                                             │
│       │     ╱  ╲     LR eleve: sauts enormes               │
│       │    ╱    ╲    → depasse l'optimum                   │
│       │   ╱      ╲   → detruit les connaissances           │
│       │  ╱   •    ╲                                        │
│       │ ╱    │     ╲  LR faible: petits pas                │
│       │╱     ↓      ╲ → converge vers l'optimum            │
│       └──────────────────► Poids                           │
│              Optimum                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Le warmup : Demarrage en douceur

Le **warmup** augmente progressivement le learning rate au debut de l'entrainement :

```python
warmup_ratio: float = 0.1  # 10% des steps en warmup
```

```
LR    ▲
      │                    ┌─────────────
5e-5  │                   ╱
      │                  ╱
      │                 ╱
      │                ╱
0     │───────────────┴─────────────────────►
      │   Warmup 10%         Training
      │                                  Steps
```

**Pourquoi ?** Les premieres iterations sont critiques. Un LR trop eleve des le debut peut destabiliser les representations pre-apprises. Le warmup permet une adaptation progressive.

---

## 6. Peu de donnees necessaires grace au pre-entrainement

### Comparaison des besoins en donnees

| Approche | Donnees necessaires | Performance attendue |
|----------|--------------------|--------------------|
| Modele from scratch | ~1 million phrases | F1 ~70% |
| Fine-tuning CamemBERT | **10,000 phrases** | **F1 ~99%** |
| Zero-shot (sans entrainement) | 0 phrase | F1 ~40% |

### Pourquoi si peu de donnees suffisent ?

Le modele pre-entraine possede deja :

1. **Connaissances lexicales** : Il sait que "Paris", "Lyon", "Marseille" sont des noms propres (probablement des lieux)

2. **Connaissances syntaxiques** : Il comprend les patterns "de X a Y", "depuis X vers Y"

3. **Connaissances contextuelles** : Il sait que dans "Je veux aller de Paris a Lyon", les mots apres "de" et "a" sont lies au deplacement

Le fine-tuning n'a qu'a apprendre :
- Quels noms propres sont des **villes** (pas des personnes)
- Comment les **classifier** en DEPART, ARRIVEE, VIA

### Illustration avec un exemple

```
Phrase: "Je veux aller de Paris a Lyon en passant par Dijon"

Ce que CamemBERT sait DEJA (pre-entrainement):
- "Paris", "Lyon", "Dijon" = noms propres (entites)
- "de X a Y" = pattern de deplacement
- "en passant par Z" = etape intermediaire

Ce que le fine-tuning apprend:
- Paris (apres "de") → B-DEPART
- Lyon (apres "a") → B-ARRIVEE
- Dijon (apres "par") → B-VIA
```

---

## 7. Implementation dans notre projet

### Configuration complete

```python
# src/nlp/camembert_ner.py

@dataclass
class TrainingConfig:
    """Configuration pour l'entrainement CamemBERT."""

    model_name: str = "camembert-base"  # Modele de base
    max_length: int = 128               # Longueur max des sequences
    batch_size: int = 16                # Taille des batches
    learning_rate: float = 5e-5         # LR faible pour fine-tuning
    num_epochs: int = 10                # Nombre d'epoques max
    warmup_ratio: float = 0.1           # 10% warmup
    weight_decay: float = 0.01          # Regularisation L2
    eval_steps: int = 100               # Evaluation toutes les 100 steps
    save_steps: int = 100               # Sauvegarde toutes les 100 steps
    early_stopping_patience: int = 3    # Arret apres 3 evals sans progres
    fp16: bool = torch.cuda.is_available()  # Mixed precision si GPU
```

### Processus d'entrainement

```python
def train(self, train_path, eval_path, output_dir):
    # 1. Charger le tokenizer pre-entraine
    self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")

    # 2. Charger le modele avec nouvelle tete de classification
    self.model = AutoModelForTokenClassification.from_pretrained(
        "camembert-base",
        num_labels=7,  # Nos 7 labels NER
    )

    # 3. Configurer l'entrainement avec LR faible
    training_args = TrainingArguments(
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    # 4. Entrainer avec early stopping
    trainer = Trainer(
        model=self.model,
        callbacks=[EarlyStoppingCallback(patience=3)],
    )
    trainer.train()
```

### Resultat

Avec seulement **10,000 phrases** et **~30 minutes** d'entrainement :

| Metrique | Score |
|----------|-------|
| F1 DEPART | 99.82% |
| F1 ARRIVEE | 99.52% |
| F1 macro | **99.67%** |

---

## 8. Bonnes pratiques pour le fine-tuning

### A faire

| Pratique | Raison |
|----------|--------|
| ✓ Utiliser un LR faible (1e-5 a 5e-5) | Preserve les connaissances |
| ✓ Activer le warmup (5-10%) | Stabilise le debut |
| ✓ Utiliser l'early stopping | Evite le sur-apprentissage |
| ✓ Evaluer regulierement | Detecte la degradation |
| ✓ Sauvegarder les checkpoints | Permet de revenir en arriere |

### A eviter

| Pratique | Risque |
|----------|--------|
| ✗ LR > 1e-4 | Oubli catastrophique |
| ✗ Trop d'epoques | Sur-apprentissage |
| ✗ Batch size trop petit | Instabilite |
| ✗ Pas de validation | Pas de detection des problemes |

---

## 9. Resume visuel

```
┌─────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ CamemBERT    │    │   + Couche   │    │   Modele     │      │
│  │ Pre-entraine │ ─► │   Classif.   │ ─► │   Fine-tune  │      │
│  │ (110M params)│    │   (7 labels) │    │   NER        │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │               │
│         │                                       │               │
│         ▼                                       ▼               │
│  Connaissances:                          Capable de:            │
│  - Francais                              - Detecter villes      │
│  - Syntaxe                               - Classifier DEP/ARR   │
│  - Semantique                            - Gerer cas difficiles │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  HYPERPARAMETRES CLES                    │   │
│  │                                                          │   │
│  │  Learning Rate: 5e-5    (preservation des connaissances) │   │
│  │  Warmup: 10%            (demarrage progressif)           │   │
│  │  Weight Decay: 0.01     (regularisation)                 │   │
│  │  Early Stopping: 3      (evite sur-apprentissage)        │   │
│  │  Epochs: 10 max         (limite par early stopping)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Resultat: F1 = 99.67% avec seulement 10,000 exemples          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- Code source : `src/nlp/camembert_ner.py` - Classes `CamemBERTTrainer` et `TrainingConfig`
- Script d'entrainement : `scripts/train_camembert.py`
- Article CamemBERT : Martin et al., 2020 - "CamemBERT: a Tasty French Language Model"
- Article BERT : Devlin et al., 2019 - "BERT: Pre-training of Deep Bidirectional Transformers"
- HuggingFace Transformers : https://huggingface.co/docs/transformers
- Voir aussi : [Pourquoi ces modeles](./pourquoi_ces_modeles.md) - spaCy/CamemBERT vs LLMs generalistes
