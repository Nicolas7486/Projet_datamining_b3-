
import pandas as pd
import numpy as np
from pathlib import Path


# Chemins des fichiers
dossier_projet = Path(__file__).resolve().parents[1]
dossier_raw = dossier_projet / "data" / "raw"
dossier_resultats = dossier_projet / "data" / "processed"

dossier_resultats.mkdir(parents=True, exist_ok=True)


# Chargement des données
stations = pd.read_csv(dossier_raw / "Citycycle_stations.csv")
trajets = pd.read_csv(dossier_raw / "Citycycle_trajets.csv")
abonnes = pd.read_csv(dossier_raw / "Citycycle_abonnes.csv")

print("Dimensions initiales :")
print("Stations :", stations.shape)
print("Trajets :", trajets.shape)
print("Abonnés :", abonnes.shape)


# Nettoyage des trajets
bilan = {}

# Suppression des lignes en double
bilan["doublons_supprimes"] = trajets.duplicated().sum()
trajets = trajets.drop_duplicates().copy()


# Les distances négatives ou nulles ne sont pas valides
distances_invalides = trajets["distance_km"] <= 0
bilan["distances_invalides"] = distances_invalides.sum()
trajets.loc[distances_invalides, "distance_km"] = np.nan


# Les trajets de plus de 180 minutes sont considérés comme anormaux
durees_anormales = trajets["duree_min"] > 180
bilan["durees_anormales"] = durees_anormales.sum()
trajets.loc[durees_anormales, "duree_min"] = np.nan


# Remplacement des températures manquantes par la médiane
bilan["temperatures_manquantes"] = trajets["temperature_c"].isna().sum()
mediane = trajets["temperature_c"].median()
trajets["temperature_c"] = trajets["temperature_c"].fillna(mediane)


# Les abonnements manquants sont indiqués comme inconnus
bilan["abonnements_manquants"] = trajets["type_abonnement"].isna().sum()
trajets["type_abonnement"] = trajets["type_abonnement"].fillna("inconnu")


# Conversion de la pluie en valeurs numériques
trajets["pluie"] = (
    trajets["pluie"].str.strip().str.lower()
    .map({"oui": 1, "non": 0})
)


# Calcul de la vitesse pour les trajets exploitables
trajets["vitesse_kmh"] = (
    trajets["distance_km"] / (trajets["duree_min"] / 60)
)

bilan["vitesses_superieures_35"] = (
    trajets["vitesse_kmh"] > 35
).sum()


# Création de nouvelles colonnes
trajets["weekend"] = trajets["jour"].isin(["sam", "dim"]).astype(int)

trajets["pointe"] = trajets["heure"].isin(
    [7, 8, 9, 17, 18, 19]
).astype(int)

trajets["boucle"] = (
    trajets["station_depart"] == trajets["station_arrivee"]
)


# Ajout du type de quartier des stations de départ et d'arrivée
stations_depart = stations[
    ["station_id", "type_quartier"]
].rename(columns={
    "station_id": "station_depart",
    "type_quartier": "quartier_depart"
})

stations_arrivee = stations[
    ["station_id", "type_quartier"]
].rename(columns={
    "station_id": "station_arrivee",
    "type_quartier": "quartier_arrivee"
})

trajets = trajets.merge(
    stations_depart,
    on="station_depart",
    how="left"
)

trajets = trajets.merge(
    stations_arrivee,
    on="station_arrivee",
    how="left"
)


# Préparation des données des abonnés
abonnes["renouvele"] = (
    abonnes["a_renouvele"].str.strip().str.lower()
    .map({"oui": 1, "non": 0})
)


# Enregistrement des fichiers nettoyés
trajets.to_csv(dossier_resultats / "trajets_propres.csv", index=False)
stations.to_csv(dossier_resultats / "stations_propres.csv", index=False)
abonnes.to_csv(dossier_resultats / "abonnes_propres.csv", index=False)


# Affichage du bilan
bilan["trajets_avant"] = 760
bilan["trajets_apres"] = len(trajets)

print("\nBilan du nettoyage :")
for nom, valeur in bilan.items():
    print(nom, ":", valeur)

print("\nNettoyage terminé !")
