# Schema BIO pour le NER (Named Entity Recognition)

## Pourquoi un simple etiquetage "VILLE" ne suffit pas ?

Imaginons une phrase comme :

> "Je veux aller de **Saint-Pierre-des-Corps** a **La Roche-sur-Yon**"

Si on utilisait un etiquetage simple ou chaque mot est etiquete "VILLE" ou "NON-VILLE", on obtiendrait :

| Token | Etiquette simple |
|-------|------------------|
| Saint | VILLE |
| Pierre | VILLE |
| des | VILLE |
| Corps | VILLE |
| La | VILLE |
| Roche | VILLE |
| sur | VILLE |
| Yon | VILLE |

**Probleme** : Comment savoir que "Saint-Pierre-des-Corps" est UNE seule ville et non quatre villes separees ? Comment distinguer la fin d'une entite du debut d'une autre ?

C'est exactement pour resoudre ce probleme que le schema **BIO** (aussi appele IOB) a ete cree.

---

## Le Schema BIO : Begin, Inside, Outside

Le schema BIO utilise 3 prefixes pour etiqueter chaque token :

### **B = Begin (Debut)**

Le premier token d'une entite. Marque le **debut** d'une nouvelle entite.

```
"Saint" dans "Saint-Pierre-des-Corps" → B-DEPART
```

### **I = Inside (Interieur)**

Les tokens **suivants** d'une entite multi-mots. Indique que le token fait partie de l'entite precedente.

```
"Pierre", "des", "Corps" → I-DEPART, I-DEPART, I-DEPART
```

### **O = Outside (Exterieur)**

Tout token qui n'appartient a **aucune entite**. La grande majorite des tokens dans une phrase.

```
"Je", "veux", "aller", "de", "a" → O, O, O, O, O
```

---

## Exemple complet avec villes composees

Prenons la phrase :

> "Je veux aller de Saint-Pierre-des-Corps a La Roche-sur-Yon"

| Token | Label BIO |
|-------|-----------|
| Je | O |
| veux | O |
| aller | O |
| de | O |
| Saint | **B-DEPART** |
| Pierre | **I-DEPART** |
| des | **I-DEPART** |
| Corps | **I-DEPART** |
| a | O |
| La | **B-ARRIVEE** |
| Roche | **I-ARRIVEE** |
| sur | **I-ARRIVEE** |
| Yon | **I-ARRIVEE** |

Grace au schema BIO, on peut maintenant **reconstruire** les entites :
- Entite 1 (DEPART) : tokens de B-DEPART jusqu'au dernier I-DEPART → "Saint-Pierre-des-Corps"
- Entite 2 (ARRIVEE) : tokens de B-ARRIVEE jusqu'au dernier I-ARRIVEE → "La Roche-sur-Yon"

---

## Les 7 labels du projet Travel Order Resolver

Dans notre projet NLP ferroviaire, nous utilisons 7 labels definis dans `src/nlp/camembert_ner.py` :

```python
LABEL2ID = {
    "O": 0,           # Outside - token hors entite
    "B-DEPART": 1,    # Begin - debut de ville de depart
    "I-DEPART": 2,    # Inside - suite de ville de depart
    "B-ARRIVEE": 3,   # Begin - debut de ville d'arrivee
    "I-ARRIVEE": 4,   # Inside - suite de ville d'arrivee
    "B-VIA": 5,       # Begin - debut de ville intermediaire
    "I-VIA": 6,       # Inside - suite de ville intermediaire
}
```

### Pourquoi 7 labels ?

- **1 label O** : pour tous les tokens non-entites
- **2 labels par type d'entite** (B + I) × 3 types d'entites = **6 labels**
- Total : **7 labels**

Les 3 types d'entites :
1. **DEPART** : ville de depart du voyage
2. **ARRIVEE** : ville de destination
3. **VIA** : villes intermediaires / correspondances

---

## Exemple avance avec VIA

> "Je veux aller de Paris a Marseille en passant par Lyon"

| Token | Label |
|-------|-------|
| Je | O |
| veux | O |
| aller | O |
| de | O |
| Paris | **B-DEPART** |
| a | O |
| Marseille | **B-ARRIVEE** |
| en | O |
| passant | O |
| par | O |
| Lyon | **B-VIA** |

Entites extraites :
- DEPART : "Paris"
- ARRIVEE : "Marseille"
- VIA : "Lyon"

---

## Reconstruction des entites : la fonction `_extract_entities()`

Le code du projet contient une fonction cle pour reconstruire les entites a partir de la sequence de labels :

```python
def _extract_entities(sequence: np.ndarray, b_label: int, i_label: int) -> list[tuple[int, int]]:
    """Extract entity spans from a label sequence."""
    entities = []
    start = None

    for i, label in enumerate(sequence):
        if label == -100:
            # Token special (padding, [CLS], [SEP]) - ignorer
            continue

        if label == b_label:
            # Nouveau debut d'entite
            if start is not None:
                # Sauvegarder l'entite precedente
                entities.append((start, i))
            start = i

        elif label == i_label:
            # Continuation d'entite
            if start is None:
                # I sans B precedent - traiter comme debut
                start = i

        else:
            # Fin d'entite (autre label ou O)
            if start is not None:
                entities.append((start, i))
                start = None

    # Gerer la derniere entite si elle termine la sequence
    if start is not None:
        entities.append((start, len(sequence)))

    return entities
```

### Algorithme explique

1. **Parcourir la sequence** de labels token par token
2. **Quand on rencontre B** :
   - Si une entite etait en cours, la sauvegarder
   - Demarrer une nouvelle entite
3. **Quand on rencontre I** :
   - Continuer l'entite en cours
   - Si pas d'entite en cours (I orphelin), le traiter comme debut
4. **Quand on rencontre O ou autre** :
   - Si une entite etait en cours, la terminer et la sauvegarder
5. **A la fin** : sauvegarder la derniere entite si necessaire

### Gestion des cas speciaux

- **Label -100** : Dans HuggingFace Transformers, -100 indique un token special a ignorer (padding, [CLS], [SEP]). Ces tokens ne font pas partie du texte original.

- **I sans B precedent** : Theoriquement invalide selon BIO strict, mais le code le gere gracieusement en le traitant comme un debut d'entite.

---

## Variantes du schema BIO

### BIO vs IOB vs BIOES

| Schema | Labels | Description |
|--------|--------|-------------|
| **BIO/IOB2** | B, I, O | Le plus courant, utilise dans ce projet |
| IOB1 | B, I, O | B seulement si deux entites se suivent |
| BIOES/BILOU | B, I, E, S, O | Ajoute End et Single pour plus de precision |

Notre projet utilise **BIO (IOB2)** car :
- Plus simple a implementer
- Suffisant pour notre cas d'usage (entites separees par des mots comme "a", "de")
- Meilleur support dans les frameworks (spaCy, HuggingFace)

---

## Resume visuel

```
Phrase: "Aller de Aix-en-Provence a Saint-Malo"

Tokens:     Aller   de   Aix   en   Provence   a   Saint   Malo
Labels:       O     O   B-DEP I-DEP  I-DEP     O   B-ARR  I-ARR

                        └────────────┘            └─────────┘
                        Entite DEPART             Entite ARRIVEE
                        "Aix-en-Provence"         "Saint-Malo"
```

Le schema BIO permet de :
- Delimiter precisement le debut et la fin des entites
- Gerer les entites multi-mots (noms composes)
- Distinguer deux entites adjacentes grace au B qui "coupe"
- Entrainer des modeles de NER supervises efficacement
