import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    stations = pd.read_csv(RAW / "Citycycle_stations.csv")
    trajets = pd.read_csv(RAW / "Citycycle_trajets.csv")
    abonnes = pd.read_csv(RAW / "Citycycle_abonnes.csv")

    print("===== NETTOYAGE =====")
    print("Trajets avant :", len(trajets))
    print("Doublons supprimés :", trajets.duplicated().sum())
    trajets = trajets.drop_duplicates().copy()

    # Recalcul des distances non positives avec les coordonnées des stations.
    coord = stations.set_index("station_id")[["longitude_km", "latitude_km"]]
    dx = trajets["station_depart"].map(coord["longitude_km"]) - trajets["station_arrivee"].map(coord["longitude_km"])
    dy = trajets["station_depart"].map(coord["latitude_km"]) - trajets["station_arrivee"].map(coord["latitude_km"])
    distance_calculee = np.sqrt(dx**2 + dy**2)
    boucle = trajets["station_depart"].eq(trajets["station_arrivee"])
    anomalie = trajets["distance_km"].le(0) & ~boucle
    print("Distances recalculées :", anomalie.sum())
    trajets.loc[anomalie, "distance_km"] = distance_calculee.loc[anomalie]
    trajets.loc[boucle, "distance_km"] = np.nan

    # Seuil conservé du code fourni : à confirmer avec le dictionnaire des données.
    print("Durées > 180 min :", trajets["duree_min"].gt(180).sum())
    trajets.loc[trajets["duree_min"].gt(180), "duree_min"] = np.nan
    trajets.loc[trajets["duree_min"].le(0), "duree_min"] = np.nan

    print("Températures manquantes :", trajets["temperature_c"].isna().sum())
    trajets["temperature_c"] = trajets["temperature_c"].fillna(trajets["temperature_c"].median())
    print("Abonnements manquants :", trajets["type_abonnement"].isna().sum())
    trajets["type_abonnement"] = trajets["type_abonnement"].fillna("inconnu")

    print("Trajets après :", len(trajets))
    print("Valeurs manquantes stations :", stations.isna().sum().sum())
    print("Valeurs manquantes abonnés :", abonnes.isna().sum().sum())
    print("Doublons stations / abonnés :", stations.duplicated().sum(), "/", abonnes.duplicated().sum())

    trajets.to_csv(OUT / "trajets_propres.csv", index=False)
    stations.to_csv(OUT / "stations_propres.csv", index=False)
    abonnes.to_csv(OUT / "abonnes_propres.csv", index=False)
    print("Fichiers propres enregistrés dans", OUT)

if __name__ == "__main__":
    main()
