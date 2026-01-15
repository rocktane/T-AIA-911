⏺ Le Tok2VecListener avec des exemples concrets

Prenons la phrase : "Je veux aller de Marseille à Paris"

---
Le problème : partager les embeddings

Dans spaCy, le pipeline peut contenir plusieurs composants :
- tok2vec : calcule les embeddings contextualisés
- ner : reconnaît les entités (DEPART, ARRIVEE)
- tagger : (optionnel) assigne les parties du discours
- parser : (optionnel) analyse les dépendances syntaxiques

Chaque composant a besoin des embeddings. Faut-il les recalculer à chaque fois ?

┌─────────────────────────────────────────────────────────────────┐
│  APPROCHE NAÏVE (inefficace)                                    │
│                                                                 │
│  Phrase → tok2vec_1 → embeddings → NER                          │
│  Phrase → tok2vec_2 → embeddings → Tagger                       │
│  Phrase → tok2vec_3 → embeddings → Parser                       │
│                                                                 │
│  → 3× le calcul, 3× les paramètres, 3× la mémoire               │
└─────────────────────────────────────────────────────────────────┘

---
La solution : le Tok2VecListener

Le Tok2VecListener permet aux composants d'ÉCOUTER les embeddings calculés par un tok2vec partagé :

┌─────────────────────────────────────────────────────────────────┐
│  APPROCHE OPTIMISÉE                                             │
│                                                                 │
│  Phrase → tok2vec (partagé) → embeddings                        │
│                                   ↓                             │
│                    ┌──────────────┴──────────────┐              │
│                    ↓              ↓              ↓              │
│               Listener 1    Listener 2    Listener 3            │
│                  (NER)       (Tagger)      (Parser)             │
│                                                                 │
│  → 1× le calcul, les composants "écoutent" le résultat          │
└─────────────────────────────────────────────────────────────────┘

---
Fonctionnement détaillé

Quand spaCy traite notre phrase :

Étape 1 : Le tok2vec calcule les embeddings
┌─────────────────────────────────────────────────────────────────┐
│  "Je veux aller de Marseille à Paris"                           │
│                    ↓                                            │
│  tok2vec (MultiHashEmbed + MaxoutWindowEncoder)                 │
│                    ↓                                            │
│  Embeddings : 7 vecteurs de 96 dimensions                       │
│  [v_Je, v_veux, v_aller, v_de, v_Marseille, v_à, v_Paris]       │
└─────────────────────────────────────────────────────────────────┘

Étape 2 : Le NER utilise son Tok2VecListener
┌─────────────────────────────────────────────────────────────────┐
│  Le Listener "écoute" le tok2vec                                │
│                    ↓                                            │
│  Récupère les embeddings déjà calculés                          │
│                    ↓                                            │
│  TransitionBasedParser → prédictions NER                        │
│  [OUT, OUT, OUT, OUT, B-DEP, OUT, B-ARR]                        │
└─────────────────────────────────────────────────────────────────┘

Pas de recalcul ! Le NER réutilise directement les embeddings.

---
Configuration dans spaCy

Voici comment le Listener est configuré dans notre projet :

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = ${components.tok2vec.model.embed.width}  # Hérite : 96
upstream = "*"                                    # Écoute tout

Explication des paramètres :
┌─────────────────────┬───────────────────────────────────────────┐
│  Paramètre          │  Description                              │
├─────────────────────┼───────────────────────────────────────────┤
│ width = ${...}      │ Dimension héritée du tok2vec (96)         │
│ upstream = "*"      │ Écoute tous les composants amont          │
└─────────────────────┴───────────────────────────────────────────┘

Le `${components.tok2vec.model.embed.width}` est une référence dynamique : si on change la width du tok2vec, le Listener s'adapte automatiquement.

---
Le paramètre "upstream"

Le paramètre `upstream` définit QUOI écouter :

┌─────────────────────┬───────────────────────────────────────────┐
│  Valeur             │  Comportement                             │
├─────────────────────┼───────────────────────────────────────────┤
│ upstream = "*"      │ Écoute le premier tok2vec disponible      │
│ upstream = "tok2vec"│ Écoute spécifiquement "tok2vec"           │
│ upstream = null     │ Pas d'écoute, embeddings intégrés         │
└─────────────────────┴───────────────────────────────────────────┘

Dans notre cas, `"*"` signifie : "utilise le tok2vec partagé du pipeline".

---
Architecture multi-tâche

Le Listener permet des architectures multi-tâches efficaces :

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      ┌────────────────┐                         │
│                      │    Tok2Vec     │                         │
│                      │   (partagé)    │                         │
│                      └───────┬────────┘                         │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐    │
│     │ Tok2VecListener│ │ Tok2VecListener│ │ Tok2VecListener│    │
│     │     (NER)      │ │   (Tagger)     │ │   (Parser)     │    │
│     └───────┬────────┘ └───────┬────────┘ └───────┬────────┘    │
│             ▼                  ▼                  ▼             │
│     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐    │
│     │ TransitionBased│ │ Tagger Layer   │ │  Parser Layer  │    │
│     │    Parser      │ │                │ │                │    │
│     └────────────────┘ └────────────────┘ └────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Chaque tâche (NER, tagging, parsing) bénéficie des mêmes embeddings riches, sans duplication.

---
Avantages du Tok2VecListener

┌─────────────────────────────────────────────────────────────────┐
│  1. EFFICACITÉ MÉMOIRE                                          │
│     Un seul tok2vec en mémoire, pas de duplication              │
│                                                                 │
│  2. VITESSE                                                     │
│     Les embeddings sont calculés une seule fois par phrase      │
│                                                                 │
│  3. COHÉRENCE                                                   │
│     Tous les composants voient les mêmes représentations        │
│                                                                 │
│  4. TRANSFERT D'APPRENTISSAGE                                   │
│     On peut pré-entraîner le tok2vec sur beaucoup de texte      │
│     puis l'utiliser pour plusieurs tâches spécialisées          │
└─────────────────────────────────────────────────────────────────┘

---
Comparaison : avec et sans Listener

┌─────────────────────┬─────────────────────┬─────────────────────┐
│                     │   Sans Listener     │   Avec Listener     │
├─────────────────────┼─────────────────────┼─────────────────────┤
│ Paramètres tok2vec  │ Dupliqués par comp. │ Partagés            │
│ Calcul embeddings   │ 1× par composant    │ 1× total            │
│ Mémoire GPU         │ N × taille tok2vec  │ 1 × taille tok2vec  │
│ Entraînement        │ Indépendant         │ Gradient partagé    │
│ Mise à jour modèle  │ N modèles à maj.    │ 1 modèle à maj.     │
└─────────────────────┴─────────────────────┴─────────────────────┘

---
Cas particulier : notre projet

Dans notre projet, le pipeline est simple :

pipeline = ["tok2vec", "ner"]

Seulement 2 composants. Mais le Listener reste utile :

┌─────────────────────────────────────────────────────────────────┐
│  Sans Listener :                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ NER aurait son propre tok2vec intégré                       ││
│  │ → Double stockage des poids                                 ││
│  │ → Gradients non partagés pendant l'entraînement             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Avec Listener :                                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ tok2vec partagé, NER écoute                                 ││
│  │ → Un seul modèle cohérent                                   ││
│  │ → Mise à jour simplifiée                                    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

---
Gradient flow pendant l'entraînement

Le Listener permet aussi le "backpropagation" à travers les composants :

┌─────────────────────────────────────────────────────────────────┐
│  FORWARD PASS                                                   │
│  Phrase → tok2vec → embeddings → Listener → NER → prédictions   │
│                                                                 │
│  BACKWARD PASS (entraînement)                                   │
│  Erreur NER → gradients → Listener → tok2vec → mise à jour      │
│                                                                 │
│  Le tok2vec apprend des erreurs du NER !                        │
└─────────────────────────────────────────────────────────────────┘

Le tok2vec s'améliore pour produire des embeddings plus utiles pour la tâche NER spécifique.

---
★ Insight ─────────────────────────────────────
Pourquoi upstream = "*" ? Le wildcard `*` est pratique car il s'adapte automatiquement. Si on renomme le composant tok2vec, le Listener continue de fonctionner. C'est plus robuste que de hardcoder un nom.

Le Listener vs embeddings statiques : Contrairement aux embeddings pré-entraînés (word2vec, fastText), le Listener utilise des embeddings DYNAMIQUES calculés par le tok2vec pour chaque phrase. Ces embeddings sont contextualisés et adaptés à la tâche.

Architecture "siamoise" : Le partage du tok2vec est similaire aux réseaux siamois où deux branches partagent les mêmes poids. Ici, NER et potentiellement d'autres composants partagent la même représentation de base.
─────────────────────────────────────────────────
