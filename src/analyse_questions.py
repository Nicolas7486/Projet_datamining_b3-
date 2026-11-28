# Ce fichier est réservé aux analyses Q1 à Q9.
# Il lit les fichiers créés par preparation_donnes.py.
# Les calculs et les méthodes sont conservés ; les résultats sont aussi exportés.

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import roc_auc_score, mean_absolute_error, silhouette_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "resultats"
QUESTIONS_OUT = OUT / "questions"
TABLEAUX_OUT = OUT / "tableaux"

OUT.mkdir(parents=True, exist_ok=True)
QUESTIONS_OUT.mkdir(parents=True, exist_ok=True)
TABLEAUX_OUT.mkdir(parents=True, exist_ok=True)

JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]


class Tee:
    """Affiche les sorties dans le terminal et les enregistre dans un fichier."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _run_analyses():
    stations = pd.read_csv(DATA / "stations_pretes.csv")
    trajets = pd.read_csv(DATA / "trajets_prepares.csv")
    abonnes = pd.read_csv(DATA / "abonnes_prets.csv")

    # Q1 — Stations à problème et heures
    print("\n===== Q1 =====")
    stations["departs"] = stations["station_id"].map(
        trajets["station_depart"].value_counts()
    ).fillna(0)
    stations["arrivees"] = stations["station_id"].map(
        trajets["station_arrivee"].value_counts()
    ).fillna(0)
    stations["solde"] = stations["departs"] - stations["arrivees"]

    correlation_q1 = stations[
        ["interventions_mois", "departs", "solde"]
    ].corr()
    moyenne_quartier_q1 = stations.groupby("type_quartier")[
        ["interventions_mois", "departs", "arrivees"]
    ].mean().round(2)
    departs_creneau_q1 = pd.crosstab(
        trajets["quartier_depart"], trajets["creneau"]
    )
    solde_quartier_q1 = stations.groupby("type_quartier")[
        ["departs", "arrivees"]
    ].sum()
    solde_quartier_q1["solde"] = (
        solde_quartier_q1["departs"] - solde_quartier_q1["arrivees"]
    )

    print(correlation_q1.round(2))
    print(moyenne_quartier_q1)
    print("Départs par quartier et créneau :")
    print(departs_creneau_q1)
    print("Solde par quartier sur toute la période :")
    print(solde_quartier_q1)

    correlation_q1.to_csv(QUESTIONS_OUT / "q1_correlations.csv")
    moyenne_quartier_q1.to_csv(QUESTIONS_OUT / "q1_moyennes_quartier.csv")
    departs_creneau_q1.to_csv(QUESTIONS_OUT / "q1_departs_quartier_creneau.csv")
    solde_quartier_q1.to_csv(QUESTIONS_OUT / "q1_solde_quartier.csv")

    moyenne_quartier_q1["interventions_mois"].plot(kind="bar")
    plt.title("Interventions moyennes par quartier")
    plt.xlabel("Type de quartier")
    plt.ylabel("Interventions moyennes par mois")
    plt.tight_layout()
    plt.savefig(QUESTIONS_OUT / "q1_interventions_quartier.png", dpi=150)
    plt.close()
    print("Important : l'heure disponible est celle du départ, pas l'heure réelle d'arrivée.")

    # Q2 — Météo, jour ou emplacement
    print("\n===== Q2 =====")
    quartiers = sorted(stations["type_quartier"].dropna().unique())
    tab = (
        trajets.groupby(["jour", "heure", "quartier_depart"])
        .size().rename("nb_trajets").reset_index()
    )
    idx = pd.MultiIndex.from_product(
        [JOURS, range(5, 24), quartiers],
        names=["jour", "heure", "quartier_depart"]
    )
    tab = tab.set_index(
        ["jour", "heure", "quartier_depart"]
    ).reindex(idx, fill_value=0).reset_index()
    tab["jour_num"] = tab["jour"].map({j: i for i, j in enumerate(JOURS)})
    tab["quartier_num"] = tab["quartier_depart"].astype("category").cat.codes

    rf = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5, random_state=0
    )
    rf.fit(tab[["heure", "jour_num", "quartier_num"]], tab["nb_trajets"])
    importance = pd.Series(
        rf.feature_importances_,
        index=["heure", "jour", "quartier"]
    )
    print("Importance des variables :\n", importance.round(2))
    importance.rename("importance").to_csv(
        QUESTIONS_OUT / "q2_importance_variables.csv"
    )

    importance.sort_values().plot(kind="barh")
    plt.title("Importance des variables")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(QUESTIONS_OUT / "q2_importance_variables.png", dpi=150)
    plt.close()

    vitesse_pluie = trajets.groupby("pluie")["vitesse_kmh"].mean().round(2)
    pointe_pluie = trajets.groupby("pluie")["pointe"].mean().round(2)
    meteo_q2 = pd.concat(
        [vitesse_pluie.rename("vitesse_moyenne_kmh"),
         pointe_pluie.rename("part_trajets_pointe")],
        axis=1
    )
    print("Vitesse moyenne selon pluie :\n", vitesse_pluie)
    print("Part des trajets en pointe selon pluie :\n", pointe_pluie)
    meteo_q2.to_csv(QUESTIONS_OUT / "q2_comparaison_pluie.csv")
    print("Limite : les données ne contiennent que les trajets réalisés.")

    # Q3 — Profil estimatif des départs à 8h
    print("\n===== Q3 =====")
    idx = pd.MultiIndex.from_product(
        [stations["station_id"], JOURS, range(5, 24)],
        names=["station_id", "jour", "heure"]
    )
    grille = trajets.groupby(
        ["station_depart", "jour", "heure"]
    ).size()
    grille.index.names = ["station_id", "jour", "heure"]
    grille = grille.reindex(idx, fill_value=0).reset_index(name="nb_departs")
    grille = grille.merge(
        stations[["station_id", "capacite", "altitude_m", "type_quartier"]],
        on="station_id"
    )
    grille["jour_num"] = grille["jour"].map(
        {j: i for i, j in enumerate(JOURS)}
    )
    grille["quartier_num"] = grille["type_quartier"].astype("category").cat.codes
    cols = ["heure", "jour_num", "quartier_num", "capacite", "altitude_m"]

    Xtr, Xte, ytr, yte = train_test_split(
        grille[cols], grille["nb_departs"], test_size=.3, random_state=0
    )
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=20, random_state=0
    ).fit(Xtr, ytr)
    mae_modele = mean_absolute_error(yte, model.predict(Xte))
    mae_moyenne = mean_absolute_error(
        yte, np.full(len(yte), ytr.mean())
    )
    print("MAE modèle :", round(mae_modele, 4))
    print("MAE moyenne :", round(mae_moyenne, 4))
    pd.DataFrame([{
        "mae_modele": mae_modele,
        "mae_moyenne": mae_moyenne
    }]).to_csv(QUESTIONS_OUT / "q3_evaluation_modele.csv", index=False)

    demain = date.today() + timedelta(days=1)
    jour = JOURS[demain.weekday()]
    model.fit(grille[cols], grille["nb_departs"])
    pred = grille[
        (grille["jour"] == jour) & (grille["heure"] == 8)
    ].copy()
    pred["prevision"] = model.predict(pred[cols])
    total = pred["prevision"].sum()
    pred["part_%"] = 100 * pred["prevision"] / total if total else 0
    print("Jour cible :", demain, jour, "; total estimé :", round(total, 2))
    print(
        pred.sort_values("prevision", ascending=False)[
            ["station_id", "type_quartier", "prevision", "part_%"]
        ].head(10).round(3)
    )
    pred[["station_id", "type_quartier", "prevision", "part_%"]].to_csv(
        QUESTIONS_OUT / "q3_prevision_demain_8h.csv", index=False
    )
    print("Limite : sans dates calendaires, ce n'est pas une vraie prévision temporelle de demain.")

    # Q4 — Test en retirant successivement un jour de semaine
    print("\n===== Q4 =====")
    resultats_q4 = []
    for j in JOURS:
        tr = tab[tab["jour"] != j]
        te = tab[tab["jour"] == j]
        m = RandomForestRegressor(
            n_estimators=100, min_samples_leaf=5, random_state=0
        )
        m.fit(tr[["heure", "quartier_num"]], tr["nb_trajets"])
        e = mean_absolute_error(
            te["nb_trajets"], m.predict(te[["heure", "quartier_num"]])
        )
        eb = mean_absolute_error(
            te["nb_trajets"],
            np.full(len(te), tr["nb_trajets"].mean())
        )
        resultats_q4.append([j, e, eb])

    q4 = pd.DataFrame(
        resultats_q4,
        columns=["jour_teste", "erreur_modele", "erreur_moyenne"]
    )
    print(q4.round(3))
    q4.to_csv(QUESTIONS_OUT / "q4_evaluation_par_jour.csv", index=False)
    print("Ce test porte sur les jours de semaine, pas sur les jours fériés ou événements.")

    # Q5 — Non-renouvellement des annuels
    print("\n===== Q5 =====")
    annuels = abonnes[abonnes["formule"] == "annuel"].copy()
    annuels["depart"] = 1 - annuels["renouvele"]
    taux_non_renouvellement = annuels["depart"].mean()
    print("Taux de non-renouvellement :", round(taux_non_renouvellement, 3))

    annuels["tranche_anciennete"] = pd.cut(
        annuels["anciennete_mois"], [0, 6, 12, 24, 36, 48, 60]
    )
    q5_anciennete = annuels.groupby(
        "tranche_anciennete", observed=False
    )["depart"].agg(["count", "mean"]).round(2)
    print(q5_anciennete)
    q5_anciennete.to_csv(QUESTIONS_OUT / "q5_anciennete.csv")

    annuels["quartile_trajets"] = pd.qcut(
        annuels["trajets_par_mois"], 4, duplicates="drop"
    )
    q5_trajets = annuels.groupby(
        "quartile_trajets", observed=False
    )["depart"].agg(["count", "mean"]).round(2)
    print(q5_trajets)
    q5_trajets.to_csv(QUESTIONS_OUT / "q5_trajets.csv")

    q5_incidents = annuels.groupby("incidents_signales")[
        "depart"
    ].agg(["count", "mean"]).round(2)
    print(q5_incidents)
    q5_incidents.to_csv(QUESTIONS_OUT / "q5_incidents.csv")

    vars_ab = [
        "anciennete_mois", "trajets_par_mois",
        "distance_moy_km", "incidents_signales"
    ]
    Xa = annuels[vars_ab].replace([np.inf, -np.inf], np.nan).fillna(
        annuels[vars_ab].median()
    )
    ya = annuels["depart"]
    auc_q5 = np.nan
    if ya.nunique() == 2 and ya.value_counts().min() >= 2:
        a, b, c, d = train_test_split(
            Xa, ya, test_size=.3, random_state=0, stratify=ya
        )
        sc = StandardScaler().fit(a)
        lg = LogisticRegression(max_iter=1000).fit(sc.transform(a), c)
        auc_q5 = roc_auc_score(
            d, lg.predict_proba(sc.transform(b))[:, 1]
        )
        print("AUC :", round(auc_q5, 2))
    else:
        print("Régression logistique non calculée : effectif/classe insuffisant.")

    pd.DataFrame([{
        "taux_non_renouvellement": taux_non_renouvellement,
        "auc": auc_q5
    }]).to_csv(QUESTIONS_OUT / "q5_indicateurs.csv", index=False)
    print("L'ancienneté est descriptive ; aucune date exacte de départ n'est disponible.")

    # Q6 — Simulation de campagne
    print("\n===== Q6 =====")
    valeur_abonne = 120
    taux_succes = .4  # hypothèses, à confirmer
    tous = abonnes.copy()
    tous["depart"] = 1 - tous["renouvele"]
    tous["mensuel"] = (tous["formule"] == "mensuel").astype(int)
    vars6 = vars_ab + ["mensuel"]
    X = tous[vars6].replace([np.inf, -np.inf], np.nan).fillna(
        tous[vars6].median()
    )
    y = tous["depart"]
    lignes_q6 = []

    if y.nunique() == 2 and y.value_counts().min() >= 5:
        prob = cross_val_predict(
            LogisticRegression(max_iter=1000),
            StandardScaler().fit_transform(X),
            y, cv=5, method="predict_proba"
        )[:, 1]
        ordre = np.argsort(-prob)
        cumul = np.cumsum(y.to_numpy()[ordre])
        total = int(y.sum())
        if total:
            n = int(np.argmax(cumul >= .5 * total) + 1)
            print(
                "Abonnés :", len(tous), "; départs :", total,
                "; contacts pour repérer 50% :", n
            )
            for cout in [2, 5, 10, 15, 20, 30]:
                gains = (
                    taux_succes * valeur_abonne * cumul
                    - cout * np.arange(1, len(cumul) + 1)
                )
                meilleur = int(np.argmax(gains) + 1)
                gain_net = float(gains.max())
                print(
                    "Coût/contact", cout, "€ : contacts", meilleur,
                    "; gain net simulé", round(gain_net, 2), "€"
                )
                lignes_q6.append({
                    "cout_contact_eur": cout,
                    "contacts_optimaux": meilleur,
                    "gain_net_simule_eur": round(gain_net, 2),
                    "valeur_abonne_hypothese_eur": valeur_abonne,
                    "taux_succes_hypothese": taux_succes
                })
            print("Hypothèses de simulation : valeur abonné = 120 €, succès = 40 %.")
        else:
            print("Aucun non-renouvellement observé.")
    else:
        print("Q6 non calculée : classes ou effectifs insuffisants.")

    pd.DataFrame(lignes_q6).to_csv(
        QUESTIONS_OUT / "q6_simulation_rentabilite.csv", index=False
    )

    # Q7 — Typologie des stations
    print("\n===== Q7 =====")
    stations["trafic"] = stations["departs"] + stations["arrivees"]
    v = ["interventions_mois", "capacite", "trafic"]
    Z = StandardScaler().fit_transform(
        stations[v].fillna(stations[v].median())
    )
    inerties = []
    sil = []
    lignes_q7_scores = []
    for k in range(2, min(9, len(stations))):
        km_test = KMeans(
            n_clusters=k, n_init=20, random_state=0
        ).fit(Z)
        inerties.append(km_test.inertia_)
        sil.append(silhouette_score(Z, km_test.labels_))
        print(
            "k =", k, "inertie =", round(km_test.inertia_, 1),
            "silhouette =", round(sil[-1], 2)
        )
        lignes_q7_scores.append({
            "k": k,
            "inertie": km_test.inertia_,
            "silhouette": sil[-1]
        })

    pd.DataFrame(lignes_q7_scores).to_csv(
        QUESTIONS_OUT / "q7_scores_clustering.csv", index=False
    )

    k_final = 4 if len(stations) >= 4 else 2
    km = KMeans(n_clusters=k_final, n_init=50, random_state=0).fit(Z)
    stations["type_station"] = km.labels_

    q7_resume = stations.groupby("type_station").agg(
        nb_stations=("station_id", "count"),
        interventions=("interventions_mois", "mean"),
        capacite=("capacite", "mean"),
        trafic=("trafic", "mean")
    ).round(1)
    q7_quartiers = pd.crosstab(
        stations["type_station"], stations["type_quartier"]
    )
    print(q7_resume)
    print(q7_quartiers)
    q7_resume.to_csv(QUESTIONS_OUT / "q7_resume_clusters.csv")
    q7_quartiers.to_csv(QUESTIONS_OUT / "q7_clusters_par_quartier.csv")
    stations.to_csv(DATA / "stations_avec_types.csv", index=False)

    # Q8 — Rééquilibrage par type (solde sur toute la période)
    print("\n===== Q8 =====")
    mapping = stations.set_index("station_id")["type_station"]
    trajets["type_depart"] = trajets["station_depart"].map(mapping)
    trajets["type_arrivee"] = trajets["station_arrivee"].map(mapping)
    dep = trajets["type_depart"].value_counts().sort_index()
    arr = trajets["type_arrivee"].value_counts().sort_index()
    solde = pd.DataFrame({"departs": dep, "arrivees": arr}).fillna(0)
    solde["solde"] = solde["departs"] - solde["arrivees"]
    print(solde)
    solde.to_csv(QUESTIONS_OUT / "q8_soldes_par_type_station.csv")
    print("Pas de solde horaire fiable : l'heure réelle d'arrivée n'est pas fournie.")

    # Q9 — Carte et zones candidates
    print("\n===== Q9 =====")
    plt.figure(figsize=(7, 6))
    plt.scatter(
        stations["longitude_km"], stations["latitude_km"],
        c=stations["solde"], cmap="RdBu_r", s=60
    )
    plt.colorbar(label="Départs - arrivées")
    plt.title("Carte des stations")
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.tight_layout()
    plt.savefig(QUESTIONS_OUT / "q9_carte_departs_arrivees.png", dpi=150)
    plt.close()

    xmin, xmax = stations["longitude_km"].min(), stations["longitude_km"].max()
    ymin, ymax = stations["latitude_km"].min(), stations["latitude_km"].max()
    xb = np.arange(np.floor(xmin), np.ceil(xmax) + 1, 1)
    yb = np.arange(np.floor(ymin), np.ceil(ymax) + 1, 1)
    if len(xb) < 2:
        xb = np.array([xmin - .5, xmax + .5])
    if len(yb) < 2:
        yb = np.array([ymin - .5, ymax + .5])

    nst, _, _ = np.histogram2d(
        stations["longitude_km"], stations["latitude_km"],
        bins=[xb, yb]
    )
    co = stations.set_index("station_id")[["longitude_km", "latitude_km"]]
    x1 = trajets["station_depart"].map(co["longitude_km"]).to_numpy()
    y1 = trajets["station_depart"].map(co["latitude_km"]).to_numpy()
    x2 = trajets["station_arrivee"].map(co["longitude_km"]).to_numpy()
    y2 = trajets["station_arrivee"].map(co["latitude_km"]).to_numpy()
    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    passages, _, _ = np.histogram2d(midx, midy, bins=[xb, yb])

    candidats = []
    for i in range(len(xb) - 1):
        for j in range(len(yb) - 1):
            if nst[i, j] == 0 and passages[i, j] >= 10:
                candidats.append([
                    (xb[i] + xb[i + 1]) / 2,
                    (yb[j] + yb[j + 1]) / 2,
                    int(passages[i, j])
                ])

    candidats = pd.DataFrame(
        candidats, columns=["x_km", "y_km", "nb_passages"]
    )
    if len(candidats):
        candidats = candidats.sort_values(
            "nb_passages", ascending=False
        )
    print("Zones candidates exploratoires :\n", candidats.head(8))
    candidats.to_csv(
        QUESTIONS_OUT / "q9_stations_a_ouvrir.csv", index=False
    )

    stations["trajets_par_place"] = np.where(
        stations["capacite"] > 0,
        stations["trafic"] / stations["capacite"],
        np.nan
    )
    stations_sollicitees = stations.sort_values(
        "trajets_par_place", ascending=False
    )[[
        "station_id", "type_quartier", "capacite",
        "trafic", "trajets_par_place"
    ]].head(8).round(2)
    print("Stations les plus sollicitées par place :")
    print(stations_sollicitees)
    stations_sollicitees.to_csv(
        QUESTIONS_OUT / "q9_stations_plus_sollicitees.csv", index=False
    )

    print("\nFin des analyses Q1 à Q9.")
    print("Résultats enregistrés dans", QUESTIONS_OUT)
    print("Rapport texte enregistré dans", OUT / "rapport_analyses.txt")


def main():
    rapport_path = OUT / "rapport_analyses.txt"
    with open(rapport_path, "w", encoding="utf-8") as rapport:
        original_stdout = sys.stdout
        try:
            sys.stdout = Tee(original_stdout, rapport)
            _run_analyses()
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
