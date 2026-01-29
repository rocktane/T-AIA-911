# Tokenization BPE (Byte-Pair Encoding)

Comment CamemBERT découpe le texte en sous-mots pour traiter n'importe quel texte français.

---

## Le problème du vocabulaire ouvert (OOV)

Un modèle classique a un vocabulaire fixe. Que faire quand il rencontre un mot inconnu ?

```
Vocabulaire limité : ["Paris", "Lyon", "Marseille", "aller", "de", "à", ...]

Phrase : "Je veux aller de Saint-Pierre-des-Corps à Châteauroux"
                         ↑                        ↑
                    Mot inconnu (OOV)         Mot inconnu (OOV)
```

Solutions historiques (toutes mauvaises) :
┌──────────────────┬────────────────────────────────────────────┐
│ Approche         │ Problème                                   │
├──────────────────┼────────────────────────────────────────────┤
│ Token <UNK>      │ Perd toute l'information du mot            │
│ Vocabulaire géant│ Mémoire énorme, mots rares mal appris      │
│ Caractère par    │ Séquences trop longues, perd le sens       │
│ caractère        │                                            │
└──────────────────┴────────────────────────────────────────────┘

Solution moderne : **BPE** — découper les mots en sous-unités fréquentes.

---

## Comment BPE découpe les mots

L'algorithme BPE apprend les paires de caractères les plus fréquentes dans un corpus, puis les fusionne itérativement.

### Construction du vocabulaire (à l'entraînement)

```
Corpus d'entraînement : "Paris, Parisien, partir, partout, ..."

Itération 1 : Les paires les plus fréquentes
  ("P", "a") apparaît 1000 fois → créer "Pa"
  ("i", "s") apparaît 800 fois  → créer "is"

Itération 2 : Sur le nouveau corpus
  ("Pa", "r") apparaît 600 fois → créer "Par"
  ("is", "ien") apparaît 400 fois → créer "isien"

... après 32000 itérations ...

Vocabulaire final : ["▁", "Par", "is", "ien", "▁de", "▁à", "Saint", "-", ...]
```

### Application (à l'inférence)

```
Entrée : "Saint-Pierre-des-Corps"

Découpage BPE :
┌───────────────────────────────────────────────────────────────┐
│  "Saint-Pierre-des-Corps"                                     │
│       ↓                                                       │
│  ["▁Saint", "-", "Pierre", "-", "des", "-", "Corps"]          │
│       ↓                                                       │
│  [  234,    45,    1892,   45,   567,   45,   3421 ]  (IDs)   │
└───────────────────────────────────────────────────────────────┘
```

---

## Le préfixe "▁" (SentencePiece)

CamemBERT utilise SentencePiece qui encode les espaces avec le caractère spécial `▁` (U+2581).

```
Phrase : "de Paris à Lyon"
          ↓
Tokens : ["▁de", "▁Paris", "▁à", "▁Lyon"]
              ↑
          Le ▁ indique "ce token commence un nouveau mot"
```

### Pourquoi c'est important ?

```
Sans espace encodé :
  "departir"  → ["de", "partir"]
  "de partir" → ["de", "partir"]   ← AMBIGUÏTÉ !

Avec SentencePiece :
  "departir"  → ["▁de", "partir"]
  "de partir" → ["▁de", "▁partir"]  ← DIFFÉRENCIABLE !
```

### Cas particuliers

┌────────────────────┬─────────────────────────────────────────┐
│ Texte              │ Tokens                                  │
├────────────────────┼─────────────────────────────────────────┤
│ "Paris"            │ ["▁Paris"]                              │
│ " Paris"           │ ["▁", "Paris"]                          │
│ "Marseille"        │ ["▁Marseille"]                          │
│ "marseille"        │ ["▁mar", "se", "ille"]  ← minuscule!    │
│ "Saint-Étienne"    │ ["▁Saint", "-", "É", "tienne"]          │
│ "aujourd'hui"      │ ["▁aujourd", "'", "hui"]                │
└────────────────────┴─────────────────────────────────────────┘

---

## Exemple concret : "Saint-Pierre-des-Corps"

Voyons le découpage complet d'un nom de gare complexe :

```
Entrée : "Saint-Pierre-des-Corps"

┌─────────────────────────────────────────────────────────────────┐
│  Token      │  ID    │  Position caractères  │  Type           │
├─────────────┼────────┼───────────────────────┼─────────────────┤
│  <s>        │  0     │  (0, 0)               │  Token spécial  │
│  ▁Saint     │  234   │  (0, 5)               │  Début de mot   │
│  -          │  45    │  (5, 6)               │  Ponctuation    │
│  Pierre     │  1892  │  (6, 12)              │  Continuation   │
│  -          │  45    │  (12, 13)             │  Ponctuation    │
│  des        │  567   │  (13, 16)             │  Continuation   │
│  -          │  45    │  (16, 17)             │  Ponctuation    │
│  Corps      │  3421  │  (17, 22)             │  Continuation   │
│  </s>       │  2     │  (0, 0)               │  Token spécial  │
└─────────────┴────────┴───────────────────────┴─────────────────┘

Total : 1 mot de 22 caractères → 9 tokens (dont 2 spéciaux)
```

---

## L'offset_mapping : aligner tokens ↔ caractères

Le problème crucial pour le NER : comment associer les prédictions par token au texte original ?

### Dans le code (`NERDataset.__getitem__()`)

```python
# Tokenize avec offset_mapping
encoding = self.tokenizer(
    text,
    max_length=self.max_length,
    padding="max_length",
    truncation=True,
    return_offsets_mapping=True,  # ← CRUCIAL
    return_tensors="pt",
)

# offset_mapping = [(0, 0), (0, 5), (5, 6), (6, 12), ...]
#                     ↑        ↑       ↑       ↑
#                   <s>    "Saint"   "-"   "Pierre"
```

### Comment l'alignement fonctionne

```
Phrase : "Je vais de Paris à Lyon"
Entités: DEPART=(10,15)="Paris", ARRIVEE=(18,22)="Lyon"

Tokenization :
┌─────────┬──────────────┬────────────────────────────────┐
│ Token   │ Offset       │ Label assigné                  │
├─────────┼──────────────┼────────────────────────────────┤
│ <s>     │ (0, 0)       │ -100 (ignoré)                  │
│ ▁Je     │ (0, 2)       │ O                              │
│ ▁vais   │ (3, 7)       │ O                              │
│ ▁de     │ (8, 10)      │ O                              │
│ ▁Paris  │ (10, 15)     │ B-DEPART  ← start=10=ent_start│
│ ▁à      │ (16, 17)     │ O                              │
│ ▁Lyon   │ (18, 22)     │ B-ARRIVEE ← start=18=ent_start│
│ </s>    │ (0, 0)       │ -100 (ignoré)                  │
└─────────┴──────────────┴────────────────────────────────┘
```

### Code d'alignement (simplifié)

```python
for offset in offset_mapping:
    start, end = offset

    if offset == (0, 0):
        # Token spécial (<s>, </s>)
        labels.append(-100)  # Ignoré dans la loss
    else:
        label = "O"
        for ent_start, ent_end, ent_label in entities:
            if start >= ent_start and end <= ent_end:
                # Ce token est dans l'entité
                if start == ent_start:
                    label = f"B-{ent_label}"  # Début
                else:
                    label = f"I-{ent_label}"  # Intérieur
        labels.append(label)
```

---

## Impact sur le NER : un mot = plusieurs tokens

C'est LE défi majeur de CamemBERT pour le NER.

### Exemple problématique

```
Phrase : "Je vais à Saint-Pierre-des-Corps"
                   └─────────────────────┘
                        Entité ARRIVEE

Tokenization :
┌───────────┬──────────────┬─────────────┐
│ Token     │ Offset       │ Label       │
├───────────┼──────────────┼─────────────┤
│ ▁Saint    │ (9, 14)      │ B-ARRIVEE   │  ← Début entité
│ -         │ (14, 15)     │ I-ARRIVEE   │  ← Intérieur
│ Pierre    │ (15, 21)     │ I-ARRIVEE   │  ← Intérieur
│ -         │ (21, 22)     │ I-ARRIVEE   │  ← Intérieur
│ des       │ (22, 25)     │ I-ARRIVEE   │  ← Intérieur
│ -         │ (25, 26)     │ I-ARRIVEE   │  ← Intérieur
│ Corps     │ (26, 31)     │ I-ARRIVEE   │  ← Intérieur
└───────────┴──────────────┴─────────────┘

7 tokens pour 1 seule entité !
```

### Schéma BIO (Begin, Inside, Outside)

```
B-DEPART  : Premier token d'une ville de départ
I-DEPART  : Tokens suivants de la même ville de départ
B-ARRIVEE : Premier token d'une ville d'arrivée
I-ARRIVEE : Tokens suivants de la même ville d'arrivée
O         : Token hors entité

Exemple complet :
"Je veux aller de Saint-Malo à La Rochelle"
 O   O    O    O  B-DEP I-DEP O B-ARR I-ARR
                  └──┬──┘     └───┬───┘
              "Saint-Malo"   "La Rochelle"
```

---

## Reconstruction des entités à l'inférence

Une fois les prédictions faites token par token, il faut reconstruire le texte original.

```python
def _reconstruct_entity(self, text: str, start: int, end: int) -> str:
    """Reconstruit le texte de l'entité depuis les positions."""
    return text[start:end].strip()
```

### Algorithme de reconstruction

```
Prédictions : [O, O, O, O, B-DEP, I-DEP, O, B-ARR, I-ARR]
Offsets :     [(0,2), (3,7), (8,10), (11,13), (14,19), (19,24), ...]

Parcours :
1. Token "B-DEP" → début d'entité, start=14
2. Token "I-DEP" → continuation, end=24
3. Token "O"     → fin d'entité DEPART
   → extraire text[14:24] = "Saint-Malo"

4. Token "B-ARR" → début nouvelle entité, start=27
5. Token "I-ARR" → continuation, end=38
6. Fin          → extraire text[27:38] = "La Rochelle"
```

---

## Comparaison avec Tok2Vec (spaCy)

┌────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect             │ spaCy Tok2Vec        │ CamemBERT BPE        │
├────────────────────┼──────────────────────┼──────────────────────┤
│ Unité de base      │ Mot entier           │ Sous-mot (subword)   │
│ "Saint-Malo"       │ 1 token              │ 2+ tokens            │
│ Mot inconnu        │ Hash embedding       │ Décomposé en sous-   │
│                    │                      │ mots connus          │
│ Contexte           │ Fenêtre glissante    │ Attention globale    │
│ Taille vocabulaire │ ~5000 (hash)         │ ~32000 (appris)      │
│ Alignement NER     │ Direct (1:1)         │ Offset mapping (N:1) │
└────────────────────┴──────────────────────┴──────────────────────┘

---

## Cas difficiles gérés par BPE

### Mots composés avec tirets
```
"Aix-en-Provence" → ["▁Aix", "-", "en", "-", "Provence"]
Tous les tokens intérieurs reçoivent le label I-ARRIVEE
```

### Minuscules (casse différente)
```
"PARIS"     → ["▁PARIS"]          (token connu)
"paris"     → ["▁par", "is"]      (décomposé !)
"Paris"     → ["▁Paris"]          (token connu)
```

### Caractères accentués
```
"Béziers"   → ["▁Bé", "z", "iers"]
"Nîmes"     → ["▁N", "î", "mes"]
```

### Apostrophes
```
"L'Haÿ-les-Roses" → ["▁L", "'", "Ha", "ÿ", "-", "les", "-", "Roses"]
```

---

★ Insight ─────────────────────────────────────
**Le label -100** : PyTorch ignore automatiquement les tokens avec label=-100 dans le calcul de la loss. C'est pourquoi on l'utilise pour les tokens spéciaux `<s>` et `</s>` — on ne veut pas que le modèle apprenne à les classifier.

**BPE vs caractères** : BPE trouve le juste milieu. "Montpellier" reste un seul token (fréquent), mais "Châteaubriant" est découpé car rare. Ceci permet un vocabulaire fixe de ~32000 tokens tout en gérant 100% des mots possibles.

**Performance NER** : Malgré la complexité de l'alignement multi-tokens, CamemBERT atteint 99.67% de F1 sur notre dataset — grâce à son pré-entraînement sur 138 Go de texte français qui lui donne une excellente compréhension contextuelle.
─────────────────────────────────────────────────
