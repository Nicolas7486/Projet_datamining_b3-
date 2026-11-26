import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

DOSSIER = "Sujet_4_Citycycle_extrait/Sujet_4_Citycycle/"

stations = pd.read_csv(DOSSIER + "Citycycle_stations.csv")
trajets = pd.read_csv(DOSSIER + "Citycycle_trajets.csv")
print(stations.shape, trajets.shape)

trajets = trajets.drop_duplicates()
trajets = trajets[trajets["distance_km"] > 0]
trajets = trajets[trajets["duree_min"] < 1440]
print("trajets propres :", len(trajets))

print(stations.groupby("type_quartier")["interventions_mois"].mean().round(2))

graves = stations[stations["interventions_mois"] >= 10]
print("stations >= 10 interventions :", len(graves))
print(graves["type_quartier"].value_counts())

triees = stations["interventions_mois"].sort_values(ascending=False)
top20 = triees.head(28).sum() / triees.sum()
print("top 20 % des stations =", round(top20 * 100), "% des interventions")

departs_par_heure = trajets.groupby("heure").size()
print(departs_par_heure)
departs_par_heure.plot(kind="bar")
plt.title("Départs par heure")
plt.savefig("departs_par_heure.png")
plt.show()
