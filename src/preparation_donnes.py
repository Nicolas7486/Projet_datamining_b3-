import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"

def main():
    trajets = pd.read_csv(DATA / "trajets_propres.csv")
    stations = pd.read_csv(DATA / "stations_propres.csv")
    abonnes = pd.read_csv(DATA / "abonnes_propres.csv")
    print("===== PRÉPARATION DES DONNÉES =====")

    pluie = trajets["pluie"].astype(str).str.strip().str.lower()
    trajets["pluie"] = pluie.map({"oui": 1, "non": 0})
    duree_h = trajets["duree_min"] / 60
    trajets["vitesse_kmh"] = np.where(duree_h > 0, trajets["distance_km"] / duree_h, np.nan)
    trajets["weekend"] = trajets["jour"].isin(["sam", "dim"]).astype(int)
    trajets["pointe"] = trajets["heure"].isin([7, 8, 9, 17, 18]).astype(int)

    quartiers = stations.set_index("station_id")["type_quartier"]
    trajets["quartier_depart"] = trajets["station_depart"].map(quartiers)
    trajets["quartier_arrivee"] = trajets["station_arrivee"].map(quartiers)
    trajets["creneau"] = pd.cut(
        trajets["heure"], [4, 6, 9, 14, 18, 23],
        labels=["5-6h", "7-9h", "10-14h", "15-18h", "19-23h"]
    )
    abonnes["renouvele"] = abonnes["a_renouvele"].astype(str).str.strip().str.lower().map({"oui": 1, "non": 0})

    trajets.to_csv(DATA / "trajets_prepares.csv", index=False)
    stations.to_csv(DATA / "stations_pretes.csv", index=False)
    abonnes.to_csv(DATA / "abonnes_prets.csv", index=False)
    print("Variables ajoutées : vitesse_kmh, weekend, pointe, quartiers, creneau, renouvele")
    print("Fichiers préparés enregistrés dans", DATA)

if __name__ == "__main__":
    main()
