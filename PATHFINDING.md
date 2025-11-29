# Pathfinding - Travel Order Resolver

## Objectif

Trouver un itineraire ferroviaire entre deux gares en utilisant les donnees SNCF.

---

## Representation en Graphe

### Structure

```
Graphe G = (V, E)
- V : Ensemble des gares (noeuds)
- E : Ensemble des connexions ferroviaires (aretes)
- w(e) : Poids de l'arete (distance ou temps)
```

### Exemple

```
        [Paris]
        /      \
       50km    300km
      /          \
   [Lyon]     [Lille]
     |
   200km
     |
 [Marseille]
```

---

## Implementation avec NetworkX

### Construction du Graphe

```python
# src/pathfinding/graph.py

import networkx as nx
import pandas as pd

class RailwayGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def load_from_csv(self, stations_file, connections_file):
        """
        Charger les donnees SNCF

        stations_file: CSV avec colonnes [id, nom, ville, lat, lon]
        connections_file: CSV avec colonnes [station1, station2, distance]
        """
        # Charger les gares
        stations = pd.read_csv(stations_file)
        for _, row in stations.iterrows():
            self.graph.add_node(
                row["ville"],
                nom=row["nom"],
                lat=row["lat"],
                lon=row["lon"]
            )

        # Charger les connexions
        connections = pd.read_csv(connections_file)
        for _, row in connections.iterrows():
            self.graph.add_edge(
                row["station1"],
                row["station2"],
                weight=row["distance"]
            )

    def get_stations(self):
        """Liste des gares"""
        return list(self.graph.nodes())

    def has_station(self, station):
        """Verifier si une gare existe"""
        return station in self.graph.nodes()
```

### Algorithme de Dijkstra

```python
# src/pathfinding/search.py

import networkx as nx

class PathFinder:
    def __init__(self, graph):
        self.graph = graph.graph

    def find_path(self, departure, destination):
        """
        Trouver le chemin le plus court entre deux gares

        Returns:
            list: [departure, step1, step2, ..., destination]
            None: si pas de chemin
        """
        try:
            path = nx.dijkstra_path(
                self.graph,
                departure,
                destination,
                weight="weight"
            )
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def find_path_with_distance(self, departure, destination):
        """
        Trouver le chemin avec la distance totale

        Returns:
            tuple: (path, total_distance)
        """
        try:
            path = nx.dijkstra_path(self.graph, departure, destination)
            distance = nx.dijkstra_path_length(self.graph, departure, destination)
            return path, distance
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, None
```

---

## Alternative : A* (si coordonnees disponibles)

```python
def heuristic(node1, node2, graph):
    """Distance euclidienne comme heuristique"""
    lat1, lon1 = graph.nodes[node1]["lat"], graph.nodes[node1]["lon"]
    lat2, lon2 = graph.nodes[node2]["lat"], graph.nodes[node2]["lon"]

    # Distance approximative en km
    return ((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) ** 0.5 * 111

def find_path_astar(graph, departure, destination):
    """A* avec heuristique geographique"""
    return nx.astar_path(
        graph,
        departure,
        destination,
        heuristic=lambda n1, n2: heuristic(n1, n2, graph),
        weight="weight"
    )
```

---

## Format de Sortie

### Entree
```
sentenceID,Departure,Destination
1,Paris,Marseille
2,Lyon,Bordeaux
```

### Sortie
```
sentenceID,Departure,Step1,Step2,...,Destination
1,Paris,Lyon,Marseille
2,Lyon,Toulouse,Bordeaux
```

### Code de formatage

```python
def format_output(sentence_id, path):
    """Formater le resultat pour la sortie"""
    if path is None:
        return f"{sentence_id},NO_PATH"
    return f"{sentence_id},{','.join(path)}"
```

---

## Donnees SNCF

### Sources

1. **Liste des gares** :
   - https://ressources.data.sncf.com/explore/dataset/liste-des-gares/

2. **Referentiel des lignes** :
   - https://ressources.data.sncf.com/explore/dataset/formes-des-lignes-du-rfn/

### Preprocessing des donnees

```python
# scripts/prepare_graph_data.py

import pandas as pd

def prepare_stations(input_file, output_file):
    """Preparer le fichier des gares"""
    df = pd.read_csv(input_file, sep=";")

    # Garder les colonnes utiles
    stations = df[["code_uic", "libelle", "commune", "latitude", "longitude"]]
    stations.columns = ["id", "nom", "ville", "lat", "lon"]

    # Nettoyer
    stations = stations.dropna()
    stations["ville"] = stations["ville"].str.strip()

    stations.to_csv(output_file, index=False)

def create_connections(stations_file, output_file):
    """
    Creer les connexions entre gares
    Note: Sans donnees officielles, on peut creer un graphe simplifie
    """
    # Connexions principales (a completer avec donnees reelles)
    connections = [
        ("Paris", "Lyon", 465),
        ("Paris", "Lille", 225),
        ("Paris", "Bordeaux", 585),
        ("Lyon", "Marseille", 315),
        ("Lyon", "Bordeaux", 550),
        ("Bordeaux", "Toulouse", 245),
        ("Toulouse", "Marseille", 405),
        # ... autres connexions
    ]

    df = pd.DataFrame(connections, columns=["station1", "station2", "distance"])
    df.to_csv(output_file, index=False)
```

---

## Complexite Algorithmique

| Algorithme | Complexite | Utilisation |
|------------|------------|-------------|
| Dijkstra | O((V + E) log V) | Chemin le plus court, poids positifs |
| A* | O(E) en moyenne | Chemin le plus court avec heuristique |
| BFS | O(V + E) | Chemin avec moins d'arrets (non pondere) |

**V** = nombre de gares (~3000 en France)
**E** = nombre de connexions

Pour notre cas, Dijkstra est suffisant et simple a implementer avec NetworkX.

---

## Tests

```python
# tests/test_pathfinding.py

def test_direct_path():
    graph = RailwayGraph()
    graph.load_from_csv("data/stations.csv", "data/connections.csv")
    finder = PathFinder(graph)

    path = finder.find_path("Paris", "Lyon")
    assert path is not None
    assert path[0] == "Paris"
    assert path[-1] == "Lyon"

def test_no_path():
    # Tester avec une gare inexistante
    path = finder.find_path("Paris", "VilleInexistante")
    assert path is None

def test_same_city():
    path = finder.find_path("Paris", "Paris")
    assert path == ["Paris"]
```
