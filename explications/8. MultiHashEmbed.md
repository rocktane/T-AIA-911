⏺ Le MultiHashEmbed avec des exemples concrets

Prenons la phrase : "Je veux aller de Marseille à Paris"

---
Qu'est-ce qu'un embedding ?

Avant de comprendre le MultiHashEmbed, rappelons ce qu'est un embedding : c'est une façon de représenter un mot par une liste de nombres (un vecteur). Ces nombres capturent le "sens" du mot d'une manière que l'ordinateur peut manipuler.

Exemple simplifié :
┌─────────────┬─────────────────────────────────┐
│    Mot      │    Embedding (simplifié)        │
├─────────────┼─────────────────────────────────┤
│ Marseille   │ [0.82, -0.15, 0.43, ...]        │
│ Paris       │ [0.79, -0.12, 0.51, ...]        │
│ aller       │ [-0.23, 0.67, -0.89, ...]       │
└─────────────┴─────────────────────────────────┘

Les villes (Marseille, Paris) ont des embeddings proches car elles partagent des caractéristiques similaires.

---
Le problème des embeddings classiques

L'approche classique utilise une table de correspondance (lookup table) :

APPROCHE CLASSIQUE (lookup table)
┌─────────────────────────────────────────────────────────────────┐
│  Vocabulaire : 500,000 mots français                            │
│  × 300 dimensions par mot                                       │
│  = 150 millions de paramètres                                   │
│  ≈ 600 Mo en mémoire                                            │
└─────────────────────────────────────────────────────────────────┘

Problèmes :
1. Taille énorme (600 Mo pour le français)
2. Mots inconnus → vecteur nul (pas d'information)
3. Fautes d'orthographe → non reconnu

---
La solution : MultiHashEmbed

Au lieu d'une grande table, on utilise 4 petites tables basées sur des ATTRIBUTS du mot :

┌─────────────────────────────────────────────────────────────────┐
│  MOT: "Marseille"                                               │
├─────────────────────────────────────────────────────────────────┤
│  ATTRIBUT   │  VALEUR        │  TABLE          │  TAILLE       │
├─────────────┼────────────────┼─────────────────┼───────────────┤
│  NORM       │  "marseille"   │  Table 1        │  5000 entrées │
│  PREFIX     │  "Mar"         │  Table 2        │  2500 entrées │
│  SUFFIX     │  "lle"         │  Table 3        │  2500 entrées │
│  SHAPE      │  "Xxxxx"       │  Table 4        │  2500 entrées │
└─────────────┴────────────────┴─────────────────┴───────────────┘

Total : 5000 + 2500 + 2500 + 2500 = 12,500 entrées
× 96 dimensions = 1.2 million de paramètres ≈ 5 Mo

C'est 120× plus petit qu'une lookup table classique !

---
Les 4 attributs expliqués

Analysons plusieurs mots de notre phrase :

┌───────────┬───────────┬────────┬────────┬───────┐
│    Mot    │   NORM    │ PREFIX │ SUFFIX │ SHAPE │
├───────────┼───────────┼────────┼────────┼───────┤
│ Je        │ je        │ Je     │ Je     │ Xx    │
│ veux      │ veux      │ veu    │ eux    │ xxxx  │
│ aller     │ aller     │ all    │ ler    │ xxxx  │
│ de        │ de        │ de     │ de     │ xx    │
│ Marseille │ marseille │ Mar    │ lle    │ Xxxxx │
│ à         │ à         │ à      │ à      │ x     │
│ Paris     │ paris     │ Par    │ ris    │ Xxxxx │
└───────────┴───────────┴────────┴────────┴───────┘

Observations clés :
- NORM : forme minuscule, utile pour identifier le mot exact
- PREFIX : "Mar" et "Par" → préfixes typiques de villes
- SUFFIX : "lle" (Marseille, Lille), "ris" (Paris) → terminaisons de villes
- SHAPE : "Xxxxx" = majuscule + minuscules = nom propre !

---
Comment ça fonctionne : le hash

Le "hash" est une fonction qui transforme n'importe quel texte en un nombre :

hash("marseille") % 5000 = 2847  → Tiroir 2847 de la table NORM
hash("Mar") % 2500 = 1203        → Tiroir 1203 de la table PREFIX
hash("lle") % 2500 = 892         → Tiroir 892 de la table SUFFIX
hash("Xxxxx") % 2500 = 156       → Tiroir 156 de la table SHAPE

Chaque tiroir contient un vecteur de 96 nombres (appris pendant l'entraînement).

┌─────────────────────────────────────────────────────────────────┐
│  TABLE "NORM" (5000 tiroirs)                                    │
├─────────┬───────────────────────────────────────────────────────┤
│ Tiroir  │ Vecteur (96 dimensions)                               │
├─────────┼───────────────────────────────────────────────────────┤
│  ...    │  ...                                                  │
│  2847   │  [0.12, -0.45, 0.78, 0.23, -0.67, ...]  ← marseille   │
│  2848   │  [0.34, 0.21, -0.56, 0.89, 0.12, ...]                 │
│  ...    │  ...                                                  │
└─────────┴───────────────────────────────────────────────────────┘

---
Combinaison des 4 vecteurs

Pour obtenir l'embedding final d'un mot, on CONCATÈNE les 4 vecteurs puis on les projette :

┌─────────────────────────────────────────────────────────────────┐
│  "Marseille"                                                    │
│                                                                 │
│  NORM   → [v1, v2, ..., v96]     ─┐                             │
│  PREFIX → [v1, v2, ..., v96]      │                             │
│  SUFFIX → [v1, v2, ..., v96]      ├→ Concaténation → 384 dim    │
│  SHAPE  → [v1, v2, ..., v96]     ─┘                             │
│                                                                 │
│  384 dimensions → Projection linéaire → 96 dimensions           │
│                                                                 │
│  Résultat : embedding final de 96 dimensions                    │
└─────────────────────────────────────────────────────────────────┘

---
La magie : les mots inconnus

Imaginons une ville que le modèle n'a jamais vue : "Pontarlier"

┌─────────────────────────────────────────────────────────────────┐
│  MOT: "Pontarlier" (jamais vu à l'entraînement)                 │
├─────────────────────────────────────────────────────────────────┤
│  NORM   = "pontarlier"    → hash vers un tiroir (collision OK)  │
│  PREFIX = "Pon"           → similaire à "Pontoise", "Pontivy"   │
│  SUFFIX = "ier"           → similaire à "Poitiers", "Montpellier│
│  SHAPE  = "Xxxxx"         → comme toutes les villes !           │
└─────────────────────────────────────────────────────────────────┘

Même si NORM pointe vers un tiroir "partagé" avec d'autres mots rares, les 3 autres attributs donnent assez d'information pour reconnaître une ville.

Comparaison avec l'approche classique :
┌─────────────────┬─────────────────────────────────────────────┐
│   Approche      │   "Pontarlier" (mot inconnu)                │
├─────────────────┼─────────────────────────────────────────────┤
│ Lookup table    │ → Vecteur nul, aucune information           │
│ MultiHashEmbed  │ → Vecteur informatif grâce à PREFIX/SUFFIX  │
└─────────────────┴─────────────────────────────────────────────┘

---
Les collisions de hash

Quand deux mots différents pointent vers le même tiroir, c'est une "collision" :

hash("marseille") % 5000 = 2847
hash("zygomatique") % 5000 = 2847  ← Collision !

Est-ce un problème ? Pas vraiment, car :
1. Les 3 autres attributs (PREFIX, SUFFIX, SHAPE) sont différents
2. Le modèle apprend à "désambiguïser" grâce au contexte
3. Les mots qui entrent en collision sont souvent rares

┌─────────────────────────────────────────────────────────────────┐
│  Même si "marseille" et "zygomatique" partagent le tiroir NORM: │
│                                                                 │
│  "marseille" : SHAPE = "Xxxxx"  → nom propre probable           │
│  "zygomatique" : SHAPE = "xxxx" → mot commun                    │
│                                                                 │
│  → Le SHAPE suffit à les distinguer                             │
└─────────────────────────────────────────────────────────────────┘

---
Configuration dans spaCy

Voici les paramètres utilisés dans notre projet :

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96                              # Dimension de sortie
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]  # 4 attributs
rows = [5000,2500,2500,2500]            # Taille de chaque table
include_static_vectors = false          # Pas de vecteurs pré-entraînés

Explication des paramètres :
┌─────────────────────┬─────────────────────────────────────────────┐
│  Paramètre          │  Description                                │
├─────────────────────┼─────────────────────────────────────────────┤
│ width = 96          │ Taille du vecteur final par mot             │
│ rows = [5000,...]   │ Nombre de tiroirs par table de hash         │
│ include_static = F  │ On n'utilise pas fastText/word2vec          │
└─────────────────────┴─────────────────────────────────────────────┘

---
Pourquoi ce choix pour notre projet ?

Notre tâche est la reconnaissance d'entités (DEPART, ARRIVEE) sur des noms de villes françaises.

┌─────────────────────────────────────────────────────────────────┐
│  Avantages du MultiHashEmbed pour notre cas :                   │
│                                                                 │
│  1. Léger : ~5 Mo vs ~500 Mo pour fastText                      │
│  2. Généralisation : reconnaît des villes jamais vues           │
│  3. Robuste : PREFIX/SUFFIX capturent les patterns de villes    │
│  4. SHAPE discriminant : "Xxxxx" = nom propre probable          │
└─────────────────────────────────────────────────────────────────┘

Pour le NER de villes, les caractéristiques de surface (majuscule, terminaisons) sont plus importantes que le sens sémantique profond.

---
★ Insight ─────────────────────────────────────
Pourquoi 96 dimensions ? C'est un compromis entre expressivité et efficacité. Avec ~3000 gares françaises à distinguer, 96 dimensions offrent suffisamment d'espace (2^96 combinaisons possibles) tout en restant rapide à calculer.

Le hash embedding est une forme de régularisation : en forçant des mots rares à partager des tiroirs, on évite le sur-apprentissage sur des exemples peu fréquents. C'est particulièrement utile pour les noms de petites villes qui n'apparaissent que quelques fois dans le dataset.

SHAPE est le roi pour le NER : dans notre corpus, 95% des entités DEPART/ARRIVEE ont le SHAPE "Xxxxx" (majuscule initiale). Ce simple attribut permet au modèle de pré-filtrer efficacement les candidats.
─────────────────────────────────────────────────
