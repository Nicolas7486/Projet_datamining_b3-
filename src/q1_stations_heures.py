import os
import pandas as pd
import matplotlib.pyplot as plt

# Le script est dans src/, les CSV nettoyés dans data/processed/
RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOSSIER = os.path.join(RACINE, "data", "processed")

stations = pd.read_csv(os.path.join(DOSSIER, "stations_propres.csv"))
trajets = pd.read_csv(os.path.join(DOSSIER, "trajets_propres.csv"))
print(stations.shape, trajets.shape)

# Nettoyage complémentaire des trajets
trajets = trajets.drop_duplicates()
trajets = trajets[trajets["distance_km"] > 0]
trajets = trajets[trajets["duree_min"] < 1440]
print("trajets propres :", len(trajets))

# Q1 : stations à problème
print(stations.groupby("type_quartier")["interventions_mois"].mean().round(2))

graves = stations[stations["interventions_mois"] >= 10]
print("stations >= 10 interventions :", len(graves))
print(graves["type_quartier"].value_counts())

# Concentration : part portée par les 20 % de stations les plus touchées
triees = stations["interventions_mois"].sort_values(ascending=False)
n20 = int(len(stations) * 0.2)
top20 = triees.head(n20).sum() / triees.sum()
print("top 20 % des stations =", round(top20 * 100), "% des interventions")

# Q1 : heures de pointe
departs_par_heure = trajets.groupby("heure").size()
print(departs_par_heure)
departs_par_heure.plot(kind="bar")
plt.title("Départs par heure")
plt.savefig(os.path.join(RACINE, "departs_par_heure.png"))
plt.show()
