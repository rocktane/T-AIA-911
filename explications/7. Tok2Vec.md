⏺ Le Tok2Vec avec des exemples concrets

Prenons la phrase : "Je veux aller de Marseille à Paris"

---
Étape 1 : L'Embedding — Créer une "fiche d'identité" pour chaque mot

Chaque mot reçoit une fiche basée sur 4 critères :

┌─────────────────────────────────────────────────────────────────┐
│  MOT: "Marseille"                                               │
├─────────────────────────────────────────────────────────────────┤
│  NORM   = "marseille"     → cherche dans table de 5000 entrées  │
│  PREFIX = "Mar"           → cherche dans table de 2500 entrées  │
│  SUFFIX = "lle"           → cherche dans table de 2500 entrées  │
│  SHAPE  = "Xxxxx"         → cherche dans table de 2500 entrées  │
└─────────────────────────────────────────────────────────────────┘

Comparons plusieurs mots :
┌───────────┬───────────┬────────┬────────┬───────┐
│    Mot    │   NORM    │ PREFIX │ SUFFIX │ SHAPE │
├───────────┼───────────┼────────┼────────┼───────┤
│ Marseille │ marseille │ Mar    │ lle    │ Xxxxx │
├───────────┼───────────┼────────┼────────┼───────┤
│ Paris     │ paris     │ Par    │ ris    │ Xxxxx │
├───────────┼───────────┼────────┼────────┼───────┤
│ Bordeaux  │ bordeaux  │ Bor    │ aux    │ Xxxxx │
├───────────┼───────────┼────────┼────────┼───────┤
│ aller     │ aller     │ all    │ ler    │ xxxx  │
├───────────┼───────────┼────────┼────────┼───────┤
│ de        │ de        │ de     │ de     │ xx    │
└───────────┴───────────┴────────┴────────┴───────┘
Ce qu'on remarque : Les villes partagent le même SHAPE (Xxxxx = majuscule puis minuscules), ce qui aide déjà le modèle à repérer les noms propres.

---
Comment ça devient des nombres ?

Imagine des tables comme des armoires à tiroirs :

TABLE "SUFFIX" (2500 tiroirs)
┌─────────┬─────────────────────────────────┐
│ Tiroir  │ Contenu (vecteur de 96 nombres) │
├─────────┼─────────────────────────────────┤
│ "lle"   │ [0.12, -0.45, 0.78, ...]        │  ← Marseille, Lille, Belleville
│ "ris"   │ [0.34, 0.21, -0.56, ...]        │  ← Paris
│ "aux"   │ [0.67, -0.12, 0.89, ...]        │  ← Bordeaux, Meaux
│ "ler"   │ [-0.23, 0.45, 0.11, ...]        │  ← aller, parler
└─────────┴─────────────────────────────────┘

Le modèle apprend pendant l'entraînement que les mots finissant par "lle", "ris", "aux" sont souvent des villes.

---
Étape 2 : L'Encodeur — Comprendre le contexte

L'embedding seul ne suffit pas. Regarde ces deux phrases :

Phrase A: "Je veux aller de Marseille à Paris"
Phrase B: "Je veux aller de Paris à Marseille"

"Paris" a la même fiche d'identité dans les deux cas, mais :
- Phrase A → Paris = ARRIVÉE
- Phrase B → Paris = DÉPART

L'encodeur regarde les mots autour pour faire la différence :

Phrase A: ... de Marseille à [Paris] ...
                           ↑
                    Le mot AVANT est "à" → c'est une ARRIVÉE

Phrase B: ... de [Paris] à Marseille ...
              ↑
       Le mot AVANT est "de" → c'est un DÉPART

---
Les 4 couches de l'encodeur — L'effet "téléphone arabe"

Avec window_size = 1, chaque couche regarde 1 mot à gauche et 1 mot à droite.

Phrase: "Je veux aller de Marseille à Paris"
         0   1     2    3     4     5   6

COUCHE 1 — Chaque mot voit ses voisins directs
┌─────────────────────────────────────────────────┐
│ "Paris" (pos 6) voit: ["à", "Paris", ∅]         │
│ → Paris sait qu'il y a "à" juste avant          │
└─────────────────────────────────────────────────┘

COUCHE 2 — Les infos se propagent
┌─────────────────────────────────────────────────┐
│ "Paris" voit maintenant l'info de "à"           │
│ Et "à" avait vu "Marseille" à la couche 1       │
│ → Paris sait indirectement qu'il y a Marseille  │
└─────────────────────────────────────────────────┘

COUCHE 3 — Encore plus de contexte
┌─────────────────────────────────────────────────┐
│ → Paris sait qu'il y a "de" quelque part        │
└─────────────────────────────────────────────────┘

COUCHE 4 — Vision complète
┌─────────────────────────────────────────────────┐
│ → Paris connaît toute la structure de la phrase │
└─────────────────────────────────────────────────┘

C'est comme un téléphone arabe : à chaque couche, l'information des voisins se transmet, et après 4 couches, chaque mot "connaît" le contexte élargi.

---
Exemple avec une ville inconnue

Imaginons une ville que le modèle n'a jamais vue : "Pontarlier"

┌─────────────────────────────────────────────────────────────────┐
│  MOT: "Pontarlier" (jamais vu à l'entraînement)                 │
├─────────────────────────────────────────────────────────────────┤
│  NORM   = "pontarlier"    → hash vers un tiroir existant        │
│  PREFIX = "Pon"           → similaire à "Pontoise", "Pontivy"   │
│  SUFFIX = "ier"           → similaire à "Poitier", "Montpellier"│
│  SHAPE  = "Xxxxx"         → comme toutes les villes !           │
└─────────────────────────────────────────────────────────────────┘

Grâce au SHAPE et au SUFFIX, le modèle devine que c'est probablement une ville, même sans l'avoir apprise.

---
★ Insight ─────────────────────────────────────
Pourquoi 96 dimensions ? : C'est un compromis. Trop petit (32) = pas assez expressif. Trop grand (256) = plus lent et risque de sur-apprentissage. 96 est suffisant pour distinguer ~3000 gares françaises.

Le hash embedding : Quand "Pontarlier" n'existe pas dans la table, son hash (ex: hash("pontarlier") % 5000 = 2847) pointe vers le tiroir 2847. Ce tiroir contient peut-être déjà d'autres mots rares, mais grâce aux 3 autres attributs (PREFIX, SUFFIX, SHAPE), le modèle peut quand même bien représenter le mot.
─────────────────────────────────────────────────
