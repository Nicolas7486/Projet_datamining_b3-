import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "resultats"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    trajets = pd.read_csv(DATA / "trajets_propres.csv")
    stations = pd.read_csv(DATA / "stations_propres.csv")
    abonnes = pd.read_csv(DATA / "abonnes_propres.csv")
    print("===== ANALYSE EXPLORATOIRE =====")
    print("Dimensions — trajets :", trajets.shape, "| stations :", stations.shape, "| abonnés :", abonnes.shape)
    print("\\nStatistiques descriptives :")
    print(trajets[["distance_km", "duree_min", "temperature_c"]].describe().round(2))
    print("\\nRépartition pluie :")
    print(trajets["pluie"].value_counts(dropna=False, normalize=True).round(2))

    jours = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]
    par_jour = trajets["jour"].value_counts().reindex(jours, fill_value=0)
    par_heure = trajets["heure"].value_counts().sort_index()
    print("\\nNombre de trajets par jour :\\n", par_jour)
    print("\\nNombre de trajets par heure :\\n", par_heure)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    par_heure.plot(kind="bar", ax=ax[0]); ax[0].set_title("Nombre de trajets par heure")
    par_jour.plot(kind="bar", ax=ax[1]); ax[1].set_title("Nombre de trajets par jour")
    ax[1].tick_params(axis="x", rotation=0)
    plt.tight_layout(); plt.savefig(OUT / "exploration_heures_jours.png"); plt.close()

    print("\\nRépartition des types d'abonnement :\\n", trajets["type_abonnement"].value_counts(dropna=False))
    if "type_velo" in trajets.columns:
        print("\\nRépartition des types de vélo :\\n", trajets["type_velo"].value_counts(dropna=False))
    print("\\nTop 10 stations selon interventions :")
    print(stations.nlargest(10, "interventions_mois")[["station_id", "type_quartier", "interventions_mois"]].to_string(index=False))
    trafic = trajets["station_depart"].value_counts().add(trajets["station_arrivee"].value_counts(), fill_value=0)
    top = stations[["station_id", "type_quartier"]].copy()
    top["trafic"] = top["station_id"].map(trafic).fillna(0)
    print("\\nTop 10 stations les plus utilisées :\\n", top.nlargest(10, "trafic").to_string(index=False))
    print("\\nGraphique enregistré dans", OUT)

if __name__ == "__main__":
    main()


