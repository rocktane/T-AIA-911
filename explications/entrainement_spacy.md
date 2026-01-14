⏺ Fonctionnement de l'entraînement spaCy

L'entraînement se décompose en deux parties : le script Python qui orchestre le processus et le fichier de configuration qui définit l'architecture et les hyperparamètres.

---
Sommaire des concepts

Ce document fait référence à plusieurs notions détaillées dans des fichiers dédiés :

| Concept | Description | Lien |
|---------|-------------|------|
| Tok2Vec | Transformation des tokens en vecteurs | [Tok2Vec.md](./Tok2Vec.md) |
| MultiHashEmbed | Embedding par hash (NORM, PREFIX, SUFFIX, SHAPE) | [MultiHashEmbed.md](./MultiHashEmbed.md) |
| MaxoutWindowEncoder | Encodeur contextuel à fenêtre glissante | [MaxoutWindowEncoder.md](./MaxoutWindowEncoder.md) |
| TransitionBasedParser | Parser NER shift-reduce | [TransitionBasedParser.md](./TransitionBasedParser.md) |
| Tok2VecListener | Connexion tok2vec ↔ NER | [Tok2VecListener.md](./Tok2VecListener.md) |
| Adam Optimizer | Optimiseur avec warmup linéaire | [AdamOptimizer.md](./AdamOptimizer.md) |
| Dropout | Régularisation (10%) | [Dropout.md](./Dropout.md) |
| Early Stopping | Patience et arrêt anticipé | [EarlyStopping.md](./EarlyStopping.md) |
| F1-Score | Métrique d'évaluation NER | [F1Score.md](./F1Score.md) |

---
1. Le Script d'entraînement (train_spacy.py)

Le script est un wrapper autour de la commande spacy train. Il :

1. Vérifie les prérequis : existence des fichiers train.spacy et test.spacy
2. Lance l'entraînement via subprocess avec la commande :
python -m spacy train configs/spacy_config.cfg \
  --output models/spacy-ner \
  --paths.train data/train.spacy \
  --paths.dev data/test.spacy \
  --training.max_steps 5000
3. Sauvegarde deux modèles : model-best (meilleur F1) et model-last (dernier checkpoint)

Arguments CLI disponibles :
┌─────────────┬──────────────────────────┬─────────────────────────────┐
│  Argument   │          Défaut          │         Description         │
├─────────────┼──────────────────────────┼─────────────────────────────┤
│ --max-steps │ 20000                    │ Nombre maximum d'itérations │
├─────────────┼──────────────────────────┼─────────────────────────────┤
│ --gpu       │ false                    │ Utiliser le GPU (CUDA)      │
├─────────────┼──────────────────────────┼─────────────────────────────┤
│ --config    │ configs/spacy_config.cfg │ Fichier de configuration    │
└─────────────┴──────────────────────────┴─────────────────────────────┘
---
2. La Configuration (spacy_config.cfg)

C'est le cœur de l'entraînement. Voici les sections clés :

[nlp] - Configuration générale

lang = "fr"                    # Langue française
pipeline = ["tok2vec","ner"]   # Composants du pipeline
batch_size = 1000              # Tokens par batch

[components.tok2vec] - Encodage des tokens

Le tok2vec transforme les tokens en vecteurs numériques. → [Explication détaillée du Tok2Vec](./Tok2Vec.md)

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96                     # Dimension des embeddings
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]  # Attributs utilisés
rows = [5000,2500,2500,2500]   # Taille des tables de hash par attribut

→ [Explication détaillée du MultiHashEmbed](./MultiHashEmbed.md)
┌──────────┬─────────────────────────────────────────┐
│ Attribut │               Description               │
├──────────┼─────────────────────────────────────────┤
│ NORM     │ Forme normalisée du token (minuscules)  │
├──────────┼─────────────────────────────────────────┤
│ PREFIX   │ Premiers caractères (préfixe)           │
├──────────┼─────────────────────────────────────────┤
│ SUFFIX   │ Derniers caractères (suffixe)           │
├──────────┼─────────────────────────────────────────┤
│ SHAPE    │ Forme abstraite (ex: Xxxx pour "Paris") │
└──────────┴─────────────────────────────────────────┘
[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96      # Largeur du réseau
depth = 4       # 4 couches convolutionnelles
window_size = 1 # Contexte ±1 token
maxout_pieces = 3  # Activation maxout à 3 pièces

→ [Explication détaillée du MaxoutWindowEncoder](./MaxoutWindowEncoder.md)

[components.ner] - Reconnaissance d'entités

@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
hidden_width = 64    # Neurones cachés
maxout_pieces = 2    # Activation maxout
update_with_oracle_cut_size = 100  # Taille des coupes pour l'oracle

Le NER utilise un parser basé sur les transitions (shift-reduce) qui prédit des actions séquentielles :
- BEGIN : début d'entité
- IN : continuation d'entité
- LAST : fin d'entité
- OUT : hors entité

→ [Explication détaillée du TransitionBasedParser](./TransitionBasedParser.md)
→ [Explication du Tok2VecListener](./Tok2VecListener.md)

[training] - Hyperparamètres d'entraînement

dropout = 0.1           # Régularisation (10% des neurones désactivés)
patience = 1600         # Early stopping après 1600 évaluations sans amélioration
max_steps = 20000       # Maximum d'itérations
eval_frequency = 200    # Évaluation toutes les 200 steps

→ [Explication du Dropout](./Dropout.md)
→ [Explication de l'Early Stopping](./EarlyStopping.md)

[training.optimizer] - Optimiseur Adam

@optimizers = "Adam.v1"
beta1 = 0.9             # Momentum du gradient
beta2 = 0.999           # Momentum du gradient carré
L2 = 0.01               # Régularisation L2 (weight decay)
grad_clip = 1.0         # Clipping du gradient (évite l'explosion)

→ [Explication détaillée de l'optimiseur Adam](./AdamOptimizer.md)

[training.optimizer.learn_rate] - Learning rate schedule

@schedules = "warmup_linear.v1"
warmup_steps = 250      # 250 steps de montée progressive
total_steps = 20000     # Durée totale
initial_rate = 0.00005  # Learning rate initial (5e-5)

Le warmup linéaire :
1. Part de 0 et monte linéairement jusqu'à 5e-5 pendant 250 steps
2. Puis décroît linéairement jusqu'à 0 sur les 19750 steps restants

[training.score_weights] - Métrique d'optimisation

ents_f = 1.0   # Optimise uniquement le F1-score
ents_p = 0.0   # Ignore la précision seule
ents_r = 0.0   # Ignore le recall seul

→ [Explication du F1-Score](./F1Score.md)

---
★ Insight ─────────────────────────────────────
Architecture choisie : Le projet utilise MultiHashEmbed plutôt que des embeddings pré-entraînés (comme fastText). C'est plus léger (~10 Mo vs ~500 Mo) et suffisant car les entités DEPART/ARRIVEE sont principalement des noms propres dont la forme (SHAPE) et les suffixes sont discriminants.

Pourquoi TransitionBasedParser ? : C'est l'architecture historique de spaCy, très efficace pour le NER. Elle traite le texte de gauche à droite en prenant des décisions séquentielles, ce qui est rapide mais ne capture pas le contexte futur (contrairement à un transformeur bidirectionnel comme CamemBERT).

Seed fixé à 42 : Garantit la reproductibilité des résultats entre différents entraînements.
─────────────────────────────────────────────────
