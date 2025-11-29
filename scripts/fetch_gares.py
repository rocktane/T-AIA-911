#!/usr/bin/env python3
"""
Script pour récupérer toutes les gares de France via l'API SNCF
et les sauvegarder dans un fichier CSV.
"""

import urllib.request
import urllib.parse
import csv
import json
import time
from typing import Dict, List, Any

API_BASE_URL = "https://ressources.data.sncf.com/api/explore/v2.1/catalog/datasets/gares-de-voyageurs/records"
LIMIT = 100  # Limite maximale par requête
OUTPUT_FILE = "data/gares-france.csv"

def fetch_gares() -> List[Dict[str, Any]]:
    """
    Récupère toutes les gares depuis l'API SNCF avec pagination.

    Returns:
        Liste de dictionnaires contenant les données des gares
    """
    all_gares = []
    offset = 0

    print("Récupération des gares depuis l'API SNCF...")

    while True:
        # Construire l'URL avec les paramètres de pagination
        params = {
            "limit": LIMIT,
            "offset": offset
        }

        try:
            print(f"  Requête: offset={offset}, limit={LIMIT}")

            # Construire l'URL avec les paramètres
            url = f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"

            # Faire la requête
            with urllib.request.urlopen(url) as response:
                # Lire et parser le JSON (décode automatiquement les séquences Unicode)
                data = json.loads(response.read().decode('utf-8'))

            # Vérifier le nombre total de résultats (première requête)
            if offset == 0:
                total_count = data.get("total_count", 0)
                print(f"  Total de gares à récupérer: {total_count}")

            # Extraire les résultats
            results = data.get("results", [])

            if not results:
                print("  Aucun résultat supplémentaire, arrêt de la récupération.")
                break

            # Traiter chaque résultat
            # Les champs sont directement dans chaque élément de results
            for gare_data in results:
                # Extraire les champs demandés
                nom = gare_data.get("nom", "")
                libellecourt = gare_data.get("libellecourt", "")
                codeinsee = gare_data.get("codeinsee", "")

                # Extraire la position géographique
                position = gare_data.get("position_geographique", {})
                if isinstance(position, dict):
                    lon = position.get("lon", "")
                    lat = position.get("lat", "")
                else:
                    lon = ""
                    lat = ""

                all_gares.append({
                    "nom": nom,
                    "libellecourt": libellecourt,
                    "lon": lon,
                    "lat": lat,
                    "codeinsee": codeinsee
                })

            print(f"  ✓ {len(results)} gares récupérées (total: {len(all_gares)})")

            # Si on a récupéré moins que la limite, on a atteint la fin
            if len(results) < LIMIT:
                break

            offset += LIMIT

            # Petite pause pour éviter de surcharger l'API
            time.sleep(0.1)

        except urllib.error.URLError as e:
            print(f"  ✗ Erreur lors de la requête: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"  ✗ Erreur lors du parsing JSON: {e}")
            break
        except Exception as e:
            print(f"  ✗ Erreur inattendue: {e}")
            break

    return all_gares

def save_to_csv(gares: List[Dict[str, Any]], filename: str):
    """
    Sauvegarde les gares dans un fichier CSV.

    Args:
        gares: Liste de dictionnaires contenant les données des gares
        filename: Chemin vers le fichier CSV de sortie
    """
    if not gares:
        print("Aucune gare à sauvegarder.")
        return

    # Définir les colonnes du CSV
    fieldnames = ["nom", "libellecourt", "lon", "lat", "codeinsee"]

    # Écrire le CSV avec encodage UTF-8
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Écrire l'en-tête
        writer.writeheader()

        # Écrire les données
        for gare in gares:
            writer.writerow(gare)

    print(f"\n✓ {len(gares)} gares sauvegardées dans {filename}")

def main():
    """Fonction principale."""
    # Récupérer toutes les gares
    gares = fetch_gares()

    if gares:
        # Sauvegarder dans le CSV
        save_to_csv(gares, OUTPUT_FILE)
        print(f"\n✓ Terminé avec succès!")
    else:
        print("\n✗ Aucune gare récupérée.")

if __name__ == "__main__":
    main()
