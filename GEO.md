# Geolocalisation - Gare la Plus Proche

## Objectif

Permettre aux utilisateurs de chercher des trajets depuis/vers des villes sans gare ferroviaire en trouvant automatiquement la gare la plus proche.

---

## Donnees Necessaires

### 1. Liste des gares SNCF avec coordonnees

**Source** : https://ressources.data.sncf.com/explore/dataset/liste-des-gares/

```csv
# data/gares_sncf.csv
id,nom,ville,latitude,longitude
1,Paris Gare de Lyon,Paris,48.8448,2.3735
2,Marseille Saint-Charles,Marseille,43.3026,5.3806
3,Lyon Part-Dieu,Lyon,45.7606,4.8590
...
```

### 2. Liste des communes francaises avec coordonnees

**Sources** :
- https://www.data.gouv.fr/fr/datasets/communes-de-france-base-des-codes-postaux/
- https://geo.api.gouv.fr/communes

```csv
# data/villes_france.csv
code_insee,nom,code_postal,latitude,longitude
75056,Paris,75000,48.8566,2.3522
13055,Marseille,13000,43.2965,5.3698
83118,Saint-Tropez,83990,43.2727,6.6406
...
```

---

## Algorithme de Distance : Haversine

La formule de Haversine calcule la distance entre deux points sur une sphere (la Terre).

```python
# src/geo/distance.py

import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcule la distance en km entre deux points GPS.

    Args:
        lat1, lon1: Coordonnees du premier point (degres)
        lat2, lon2: Coordonnees du second point (degres)

    Returns:
        float: Distance en kilometres
    """
    R = 6371  # Rayon de la Terre en km

    # Conversion en radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    # Formule de Haversine
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c
```

---

## Recherche de la Gare la Plus Proche

### Approche Naive (O(n))

```python
# src/geo/nearest_station.py

import pandas as pd
from .distance import haversine

class NearestStationFinder:
    def __init__(self, stations_file):
        self.stations = pd.read_csv(stations_file)

    def find_nearest(self, lat, lon):
        """
        Trouve la gare la plus proche d'un point GPS.

        Returns:
            tuple: (nom_gare, distance_km)
        """
        min_distance = float('inf')
        nearest_station = None

        for _, station in self.stations.iterrows():
            distance = haversine(
                lat, lon,
                station['latitude'], station['longitude']
            )
            if distance < min_distance:
                min_distance = distance
                nearest_station = station['nom']

        return nearest_station, round(min_distance, 1)
```

### Approche Optimisee avec KD-Tree (O(log n))

```python
# src/geo/nearest_station.py

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

class NearestStationFinderOptimized:
    def __init__(self, stations_file):
        self.stations = pd.read_csv(stations_file)

        # Construire le KD-Tree avec les coordonnees
        coords = self.stations[['latitude', 'longitude']].values
        # Conversion en radians pour plus de precision
        coords_rad = np.radians(coords)
        self.tree = cKDTree(coords_rad)

        self.R = 6371  # Rayon de la Terre en km

    def find_nearest(self, lat, lon, k=1):
        """
        Trouve les k gares les plus proches.

        Args:
            lat, lon: Coordonnees du point
            k: Nombre de gares a retourner

        Returns:
            list: [(nom_gare, distance_km), ...]
        """
        point_rad = np.radians([lat, lon])

        # Recherche dans le KD-Tree
        distances_rad, indices = self.tree.query(point_rad, k=k)

        results = []
        for dist_rad, idx in zip(
            np.atleast_1d(distances_rad),
            np.atleast_1d(indices)
        ):
            station = self.stations.iloc[idx]
            # Conversion distance angulaire -> km (approximation)
            distance_km = dist_rad * self.R
            results.append((station['nom'], round(distance_km, 1)))

        return results if k > 1 else results[0]
```

---

## Integration dans le Pipeline

### Module de Resolution de Ville

```python
# src/geo/city_resolver.py

import pandas as pd
from .nearest_station import NearestStationFinderOptimized

class CityResolver:
    def __init__(self, stations_file, cities_file):
        self.stations = pd.read_csv(stations_file)
        self.cities = pd.read_csv(cities_file)
        self.finder = NearestStationFinderOptimized(stations_file)

        # Set des noms de gares pour lookup rapide
        self.station_names = set(
            self.stations['ville'].str.lower()
        )

    def resolve(self, city_name):
        """
        Resout une ville en gare.

        Args:
            city_name: Nom de la ville

        Returns:
            dict: {
                'ville_demandee': str,
                'gare': str,
                'distance_km': float,
                'est_gare': bool
            }
        """
        city_lower = city_name.lower()

        # Cas 1: La ville est une gare
        if city_lower in self.station_names:
            return {
                'ville_demandee': city_name,
                'gare': city_name,
                'distance_km': 0,
                'est_gare': True
            }

        # Cas 2: Chercher les coordonnees de la ville
        city_row = self.cities[
            self.cities['nom'].str.lower() == city_lower
        ]

        if city_row.empty:
            return None  # Ville inconnue

        lat = city_row.iloc[0]['latitude']
        lon = city_row.iloc[0]['longitude']

        # Trouver la gare la plus proche
        gare, distance = self.finder.find_nearest(lat, lon)

        return {
            'ville_demandee': city_name,
            'gare': gare,
            'distance_km': distance,
            'est_gare': False
        }
```

---

## Utilisation dans le Pipeline Principal

```python
# src/main.py

from nlp.baseline import BaselineNER
from geo.city_resolver import CityResolver
from pathfinding.search import PathFinder

def process_sentence(sentence_id, sentence, nlp, resolver, pathfinder):
    # Etape 1: Extraction NLP
    departure, destination = nlp.extract(sentence)

    if departure is None or destination is None:
        return f"{sentence_id},INVALID"

    # Etape 2: Resolution des villes en gares
    dep_info = resolver.resolve(departure)
    arr_info = resolver.resolve(destination)

    if dep_info is None or arr_info is None:
        return f"{sentence_id},UNKNOWN_CITY"

    # Etape 3: Pathfinding
    path = pathfinder.find_path(dep_info['gare'], arr_info['gare'])

    if path is None:
        return f"{sentence_id},NO_PATH"

    # Etape 4: Formatage de la sortie
    path_str = "->".join(path)

    return (
        f"{sentence_id},"
        f"{dep_info['ville_demandee']},"
        f"{dep_info['gare']},"
        f"{arr_info['ville_demandee']},"
        f"{arr_info['gare']},"
        f"\"{path_str}\""
    )
```

---

## Tests

```python
# tests/test_geo.py

def test_city_is_station():
    resolver = CityResolver("data/gares.csv", "data/villes.csv")
    result = resolver.resolve("Paris")

    assert result['est_gare'] == True
    assert result['distance_km'] == 0

def test_city_without_station():
    resolver = CityResolver("data/gares.csv", "data/villes.csv")
    result = resolver.resolve("Saint-Tropez")

    assert result['est_gare'] == False
    assert result['gare'] == "Saint-Raphael"
    assert result['distance_km'] < 50

def test_unknown_city():
    resolver = CityResolver("data/gares.csv", "data/villes.csv")
    result = resolver.resolve("VilleInexistante")

    assert result is None
```

---

## Performance

| Methode | Complexite | Temps (3000 gares) |
|---------|------------|-------------------|
| Naive (boucle) | O(n) | ~1 ms |
| KD-Tree | O(log n) | ~0.01 ms |

Pour notre cas (~3000 gares), la methode naive est acceptable.
Le KD-Tree devient utile si on fait beaucoup de requetes.

---

## Ameliorations Possibles

1. **Top-K gares** : Proposer les 3 gares les plus proches
2. **Filtrage par type** : TGV, TER, etc.
3. **Prise en compte du temps d'acces** : Distance routiere vs vol d'oiseau
4. **Cache** : Memoriser les resultats frequents
