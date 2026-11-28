import argparse
import sys
from pathlib import Path

import pandas as pd


DONNEES = Path(__file__).resolve().parents[1] / "data" / "processed"
JOURS = ("lun", "mar", "mer", "jeu", "ven", "sam", "dim")
WEEKEND = ("sam", "dim")


def prevoir_total(effectifs, jour):
    # Avec un indicateur binaire, les moindres carrés donnent les moyennes
    # des deux groupes : semaine et week-end.
    meme_type = effectifs.index.isin(WEEKEND) == (jour in WEEKEND)
    return effectifs.loc[meme_type].mean()


def estimer_departs(trajets, stations, jour):
    departs_8h = trajets.loc[
        trajets["heure"].eq(8) & trajets["jour"].isin(JOURS)
    ]
    ids_stations = stations["station_id"]
    if ids_stations.isna().any() or ids_stations.duplicated().any():
        raise ValueError("Les identifiants de station doivent être renseignés et uniques.")
    if departs_8h.empty or not departs_8h["station_depart"].isin(ids_stations).all():
        raise ValueError("Les départs de 8 h doivent être rattachés aux stations du réseau.")
    # Garder les jours présents dans le fichier, même sans départ à 8 h.
    jours_observes = [j for j in JOURS if j in set(trajets["jour"])]
    totaux = departs_8h.groupby("jour").size().reindex(jours_observes, fill_value=0)
    total_jour = prevoir_total(totaux, jour)
    effectifs = departs_8h["station_depart"].value_counts()
    # Partir de la liste des stations pour inclure celles sans départ observé à 8 h.
    resultat = stations[["station_id"]].copy()
    resultat["departs_observes_8h_tous_jours"] = ids_stations.map(effectifs).fillna(0).astype(int)
    # Répartir le total prévu selon la part de chaque station dans les départs de 8 h.
    resultat["departs_estimes_8h"] = (
        total_jour * resultat["departs_observes_8h_tous_jours"] / len(departs_8h)
    )
    resultat.insert(0, "jour_cible", jour)
    resultat = resultat.sort_values(["departs_estimes_8h", "station_id"], ascending=[False, True])
    return resultat, totaux


def comparer_erreurs(totaux):
    erreurs_regression = []
    erreurs_reference = []
    # Retirer une catégorie de jour à la fois et comparer les deux modèles sur ce même total.
    for jour, observe in totaux.items():
        entrainement = totaux.drop(jour)
        erreurs_regression.append(abs(observe - prevoir_total(entrainement, jour)))
        erreurs_reference.append(abs(observe - entrainement.mean()))
    return sum(erreurs_regression) / len(totaux), sum(erreurs_reference) / len(totaux)


def main():
    parser = argparse.ArgumentParser(description="Estimation des départs par station à 8 h.")
    parser.add_argument("jour", choices=JOURS, help="Jour de la semaine correspondant à demain.")
    parser.add_argument("--evaluation", action="store_true", help="Affiche les erreurs réseau sur stderr.")
    args = parser.parse_args()

    trajets = pd.read_csv(DONNEES / "trajets_propres.csv")
    stations = pd.read_csv(DONNEES / "stations_propres.csv")
    resultat, totaux = estimer_departs(trajets, stations, args.jour)
    resultat.to_csv(sys.stdout, index=False, lineterminator="\n")
    if args.evaluation:
        # Afficher les erreurs séparément pour que la redirection contienne uniquement le CSV.
        regression, reference = comparer_erreurs(totaux)
        print(f"Erreur absolue moyenne réseau : régression {regression:.2f}, moyenne {reference:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
