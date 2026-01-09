# Script de Presentation - Travel Order Resolver

Duree totale : ~10 minutes

---

## Slide 1 : Titre (30s)

Bonjour, je vais vous presenter notre projet **Travel Order Resolver**.

L'objectif : transformer une phrase en francais comme "Je veux aller de Paris a Lyon" en un itineraire ferroviaire complet.

Le projet repose sur 4 piliers : le NLP pour comprendre la phrase, la resolution ville vers gare, le pathfinding pour calculer l'itineraire, et les donnees SNCF.

---

## Slide 2 : Problematique (45s)

Le defi principal c'est de comprendre le **langage naturel**. Les gens s'expriment de plein de facons differentes : "Je veux aller de...", "Comment me rendre a...", "Un billet pour...".

On doit faire de l'**extraction d'entites** (NER) pour identifier les villes de depart et d'arrivee. C'est complexe car certains prenoms sont aussi des villes : Paris, Florence, Albert...

On doit aussi **distinguer les phrases valides des invalides**. "Il fait beau aujourd'hui" n'est pas un ordre de voyage, on doit retourner INVALID.

Enfin, on exploite les **donnees SNCF Open Data** pour les gares et on a genere notre propre dataset de 10 000 phrases pour entrainer nos modeles.

---

## Slide 3 : Pipeline (45s)

Voici l'architecture complete.

On recoit en entree un fichier CSV avec des phrases au format `sentenceID,sentence`.

Le **module NLP** fait le preprocessing, la reconnaissance d'entites, et la classification valide/invalide. On peut utiliser spaCy ou CamemBERT.

Ensuite, la **resolution ville vers gare** associe chaque ville extraite a une gare SNCF. Si la ville n'a pas de gare, on trouve la plus proche.

Le **pathfinding** calcule le chemin optimal sur le graphe du reseau SNCF avec Dijkstra.

En sortie : `sentenceID,Departure,Destination` ou `INVALID`.

---

## Slide 4 : Donnees & Dataset (45s)

Pour les donnees, on utilise **SNCF Open Data** : la liste des gares avec leurs coordonnees GPS et les connexions ferroviaires.

On a aussi les **35 000 communes francaises** pour pouvoir resoudre n'importe quelle ville vers sa gare la plus proche.

Pour le NLP, on a **genere 10 000 phrases annotees** a partir de 300 templates grammaticaux differents. 8000 pour l'entrainement, 2000 pour les tests. Les annotations sont au format BIO standard.

Le format d'entree/sortie est simple : CSV en UTF-8, une phrase par ligne.

---

## Slide 5 : Stack Technique (30s)

Cote technique :
- **Python 3.12** avec le gestionnaire de paquets `uv`
- **spaCy** pour le NER avec un modele custom entraine
- **CamemBERT** fine-tune pour une precision maximale
- **SciPy** pour le KD-Tree de recherche spatiale
- **NetworkX** pour l'algorithme de Dijkstra
- **FastAPI** pour l'interface web

---

## Slide 6 : Modeles NLP (1min)

On a implemente et compare **3 approches**.

Le **Baseline** avec regex et dictionnaire : 73% de F1 sur le depart, 71% sur l'arrivee. C'est notre point de reference.

**spaCy custom** entraine sur notre dataset : on monte a 95-96% de F1. C'est deja tres bien.

**CamemBERT fine-tune** : on atteint **99.8%** de F1. C'est notre meilleur modele.

Le dataset de 10 000 phrases a ete crucial. On a entraine spaCy sur 5000 steps, et CamemBERT sur 5 epochs avec early stopping.

---

## Slide 7 : Cas Difficiles (45s)

Notre modele gere plusieurs cas difficiles mentionnes dans l'enonce :

- Les **prenoms qui sont des villes** : Albert, Paris, Florence, Nancy
- Les **noms composes** : Port-Boulet, Saint-Pierre-des-Corps
- Les phrases **sans majuscules** : "je veux aller de paris a lyon"
- Les phrases **sans accents** : "de beziers a montpellier"
- Les **fautes d'orthographe** : Marseile, Bordeau - on fait du fuzzy matching
- L'**ordre inverse** : "A Marseille depuis Lyon" - l'arrivee avant le depart

---

## Slide 8 : Resolution Ville-Gare & Pathfinding (45s)

Une fois les villes extraites par le NLP, on doit les **associer a des gares**.

On fait la correspondance via le code INSEE. Si la ville n'a pas de gare directe, on utilise un **KD-Tree** pour trouver la gare la plus proche geographiquement.

Pour le **graphe SNCF**, les gares sont des noeuds et les connexions ferroviaires sont des aretes. On importe ca depuis les CSV Open Data.

**Dijkstra** calcule le plus court chemin et retourne la sequence : Depart, Etape1, Etape2, ..., Destination.

---

## Slide 9 : Chiffres Cles (30s)

Quelques chiffres :
- **99.67%** de F1 macro avec CamemBERT
- **10 000** phrases d'entrainement generees
- **3000+** gares SNCF dans notre graphe
- **35 000+** communes francaises pour la resolution

On utilise le schema **BIO** standard : B-DEPART, I-DEPART, B-ARRIVEE, I-ARRIVEE, O.

---

## Slide 10 : Bonus (45s)

On a implemente plusieurs bonus :

**Speech-to-Text** avec Whisper : reconnaissance vocale integree a l'interface web. Le modele tourne cote client en WebGPU. On maintient Espace pour dicter.

**Arrets intermediaires (VIA)** : on peut dire "De Paris a Lyon en passant par Dijon". On a etendu le NER avec des labels B-VIA/I-VIA.

**Correction orthographique** : fuzzy matching pour tolerer les fautes de frappe.

**Benchmark complet** : comparaison des 3 modeles avec toutes les metriques.

---

## Slide 11 : Demo (30s)

Passons a la demonstration.

On peut lancer l'interface web avec `uvicorn` ou utiliser le programme en ligne de commande qui lit les phrases sur stdin.

*[Lancer la demo]*

---

## Notes pour la demo

Exemples a tester :
- "Je veux aller de Paris a Lyon" (cas simple)
- "je voudrais un billet marseille bordeaux" (sans majuscules, ordre depart-arrivee implicite)
- "A Toulouse depuis Nantes en passant par Limoges" (VIA + ordre inverse)
- "Avec mon ami Albert, je vais de Nancy a Lourdes" (prenoms = villes)
- "Il fait beau aujourd'hui" (INVALID)
- Test vocal si le temps le permet
