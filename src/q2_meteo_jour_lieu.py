import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

RACINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOSSIER = os.path.join(RACINE, "data", "processed")

stations = pd.read_csv(os.path.join(DOSSIER, "stations_propres.csv"))
trajets = pd.read_csv(os.path.join(DOSSIER, "trajets_propres.csv"))

# Même nettoyage que la Q1
trajets = trajets.drop_duplicates()
trajets = trajets[trajets["distance_km"] > 0]
trajets = trajets[trajets["duree_min"] < 1440]
print("trajets propres :", len(trajets))

# Emplacement : type de quartier et altitude de la station de départ
trajets = trajets.merge(
    stations[["station_id", "type_quartier", "altitude_m"]],
    left_on="station_depart", right_on="station_id", how="left")

# ---------- 1. Analyse exploratoire ----------
print(trajets[["temperature_c", "distance_km", "altitude_m", "duree_min"]].describe())
print(trajets[["temperature_c", "distance_km", "altitude_m", "duree_min"]].corr().round(2))
print(trajets.groupby("pluie")["duree_min"].mean().round(1))
print(trajets.groupby("jour")["duree_min"].mean().round(1))
print(trajets.groupby("type_quartier")["duree_min"].mean().round(1))

# ---------- 2. Préparation : variables et cible ----------
# Cible : la durée du trajet. La distance sert de contrôle, pour ne pas
# confondre l'effet de la météo avec la longueur du trajet.
X = pd.get_dummies(
    trajets[["temperature_c", "pluie", "jour", "type_quartier",
             "altitude_m", "distance_km"]],
    drop_first=True, dtype=float)
y = trajets["duree_min"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Référence : prédire toujours la moyenne
rmse_ref = np.sqrt(mean_squared_error(y_test, np.full(len(y_test), y_train.mean())))
print("RMSE de référence (moyenne) :", round(rmse_ref, 2))

# ---------- 3. Régression linéaire (variables standardisées) ----------
# Standardiser permet de comparer le poids de chaque variable.
reg = make_pipeline(StandardScaler(), LinearRegression())
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print("Régression linéaire : RMSE =", round(np.sqrt(mean_squared_error(y_test, pred)), 2),
      "| R2 =", round(r2_score(y_test, pred), 3))

poids = pd.Series(reg[-1].coef_, index=X.columns)
print("Poids des variables (triés par importance) :")
print(poids.reindex(poids.abs().sort_values(ascending=False).index).round(2))

# ---------- 4. kNN : choix de k par validation croisée ----------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
meilleur_k, meilleur_score = None, -np.inf
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    modele = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k))
    score = cross_val_score(modele, X_train, y_train, cv=cv,
                            scoring="neg_root_mean_squared_error").mean()
    print("k =", k, "| RMSE (validation croisée) =", round(-score, 2))
    if score > meilleur_score:
        meilleur_k, meilleur_score = k, score
print("Meilleur k :", meilleur_k)

knn = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=meilleur_k))
knn.fit(X_train, y_train)
pred_knn = knn.predict(X_test)
print("kNN : RMSE =", round(np.sqrt(mean_squared_error(y_test, pred_knn)), 2),
      "| R2 =", round(r2_score(y_test, pred_knn), 3))
