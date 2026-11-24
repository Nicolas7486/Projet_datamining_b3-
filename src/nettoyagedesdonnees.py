"""Citycycle - étape 1 : nettoyage. Chaque règle est justifiée en commentaire."""
import pandas as pd
import numpy as np
D = "c:/ece/bdd/projet/"
s = pd.read_csv(D+"Citycycle_stations.csv")
t = pd.read_csv(D+"Citycycle_trajets.csv")
a = pd.read_csv(D+"Citycycle_abonnes.csv")
log = {}

# --- TRAJETS ---
log["trajets_brut"] = len(t)
# 1) lignes strictement identiques = doublons d'export (même trajet_id)
t = t.drop_duplicates().copy(); log["après doublons"] = len(t)
# 2) distance <= 0 : impossible (-1 = code d'erreur) -> NaN, la ligne reste utile pour l'heure/jour
bad_d = t.distance_km <= 0; log["distance<=0 mises à NaN"] = int(bad_d.sum())
t.loc[bad_d, "distance_km"] = np.nan
# 3) durée > 180 min : vélo non rendu / session ouverte (1440 = 24 h) -> NaN
bad_t = t.duree_min > 180; log["durée>180 mises à NaN"] = int(bad_t.sum())
t.loc[bad_t, "duree_min"] = np.nan
# 4) vitesse implicite (km/h) : contrôle de cohérence
t["vitesse_kmh"] = t.distance_km / (t.duree_min / 60)
bad_v = t.vitesse_kmh > 35; log["vitesse>35 km/h (flag)"] = int(bad_v.sum())
# 5) départ = arrivée : conservé (boucle possible) mais signalé
t["boucle"] = t.station_depart == t.station_arrivee
# 6) manquants : type d'abonnement -> 'inconnu' (catégorie, on n'invente pas) ;
#    température -> médiane par jour... non : médiane globale + indicateur
t["type_abonnement"] = t.type_abonnement.fillna("inconnu")
t["temp_manquante"] = t.temperature_c.isna()
t["temperature_c"] = t.temperature_c.fillna(t.temperature_c.median())
# 7) variables utiles
t["pluie"] = (t.pluie == "oui").astype(int)
t["weekend"] = t.jour.isin(["sam", "dim"]).astype(int)
t["pointe"] = t.heure.isin([7,8,9,17,18,19]).astype(int)

# --- STATIONS : propres. on joint le type de quartier aux trajets
t = t.merge(s[["station_id","type_quartier"]].rename(columns={"station_id":"station_depart","type_quartier":"quartier_depart"}), on="station_depart", how="left")

# --- ABONNES : variable cible binaire
a["renouvele"] = (a.a_renouvele == "oui").astype(int)

t.to_csv("trajets_propres.csv", index=False)
s.to_csv("stations_propres.csv", index=False)
a.to_csv("abonnes_propres.csv", index=False)
for k, v in log.items(): print(f"{k:32s}{v}")
print("lignes finales :", len(t), len(s), len(a))
