# Ce fichier est réservé aux analyses Q1 à Q9.
# Il lit les fichiers créés par preparation_donnees.py.
# Les analyses sont séparées du nettoyage et de l'exploration pour garder
# la même organisation que le premier travail.

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
OUT.mkdir(parents=True, exist_ok=True)
JOURS = ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"]

def main():
    stations = pd.read_csv(DATA / "stations_pretes.csv")
    trajets = pd.read_csv(DATA / "trajets_prepares.csv")
    abonnes = pd.read_csv(DATA / "abonnes_prets.csv")

    # Q1 — Stations à problème et heures
    print("\\n===== Q1 =====")
    stations["departs"] = stations["station_id"].map(trajets["station_depart"].value_counts()).fillna(0)
    stations["arrivees"] = stations["station_id"].map(trajets["station_arrivee"].value_counts()).fillna(0)
    stations["solde"] = stations["departs"] - stations["arrivees"]
    print(stations[["interventions_mois", "departs", "solde"]].corr().round(2))
    print(stations.groupby("type_quartier")[["interventions_mois", "departs", "arrivees"]].mean().round(2))
    stations.groupby("type_quartier")["interventions_mois"].mean().plot(kind="bar")
    plt.title("Interventions moyennes par quartier"); plt.tight_layout()
    plt.savefig(OUT / "q1_interventions_quartier.png"); plt.close()
    print("Départs par quartier et créneau :")
    print(pd.crosstab(trajets["quartier_depart"], trajets["creneau"]))
    print("Solde par quartier sur toute la période :")
    print(stations.groupby("type_quartier")[["departs","arrivees"]].sum().assign(
        solde=lambda x: x["departs"]-x["arrivees"]))
    print("Important : l'heure disponible est celle du départ, pas l'heure réelle d'arrivée.")

    # Q2 — Météo, jour ou emplacement
    print("\\n===== Q2 =====")
    quartiers = sorted(stations["type_quartier"].dropna().unique())
    tab = trajets.groupby(["jour","heure","quartier_depart"]).size().rename("nb_trajets").reset_index()
    idx = pd.MultiIndex.from_product([JOURS, range(5,24), quartiers], names=["jour","heure","quartier_depart"])
    tab = tab.set_index(["jour","heure","quartier_depart"]).reindex(idx, fill_value=0).reset_index()
    tab["jour_num"] = tab["jour"].map({j:i for i,j in enumerate(JOURS)})
    tab["quartier_num"] = tab["quartier_depart"].astype("category").cat.codes
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=0)
    rf.fit(tab[["heure","jour_num","quartier_num"]], tab["nb_trajets"])
    importance = pd.Series(rf.feature_importances_, index=["heure","jour","quartier"])
    print("Importance des variables :\\n", importance.round(2))
    importance.sort_values().plot(kind="barh"); plt.title("Importance des variables")
    plt.tight_layout(); plt.savefig(OUT/"q2_importance_variables.png"); plt.close()
    print("Vitesse moyenne selon pluie :\\n", trajets.groupby("pluie")["vitesse_kmh"].mean().round(2))
    print("Part des trajets en pointe selon pluie :\\n", trajets.groupby("pluie")["pointe"].mean().round(2))
    print("Limite : les données ne contiennent que les trajets réalisés.")

    # Q3 — Profil estimatif des départs à 8h
    print("\\n===== Q3 =====")
    idx = pd.MultiIndex.from_product([stations["station_id"], JOURS, range(5,24)],
                                     names=["station_id","jour","heure"])
    grille = trajets.groupby(["station_depart","jour","heure"]).size()
    grille.index.names = ["station_id","jour","heure"]
    grille = grille.reindex(idx, fill_value=0).reset_index(name="nb_departs")
    grille = grille.merge(stations[["station_id","capacite","altitude_m","type_quartier"]], on="station_id")
    grille["jour_num"] = grille["jour"].map({j:i for i,j in enumerate(JOURS)})
    grille["quartier_num"] = grille["type_quartier"].astype("category").cat.codes
    cols = ["heure","jour_num","quartier_num","capacite","altitude_m"]
    Xtr,Xte,ytr,yte = train_test_split(grille[cols],grille["nb_departs"],test_size=.3,random_state=0)
    model = RandomForestRegressor(n_estimators=200,min_samples_leaf=20,random_state=0).fit(Xtr,ytr)
    print("MAE modèle :", round(mean_absolute_error(yte,model.predict(Xte)),4))
    print("MAE moyenne :", round(mean_absolute_error(yte,np.full(len(yte),ytr.mean())),4))
    demain = date.today()+timedelta(days=1); jour = JOURS[demain.weekday()]
    model.fit(grille[cols],grille["nb_departs"])
    pred = grille[(grille["jour"]==jour)&(grille["heure"]==8)].copy()
    pred["prevision"] = model.predict(pred[cols])
    total = pred["prevision"].sum()
    pred["part_%"] = 100*pred["prevision"]/total if total else 0
    print("Jour cible :",demain,jour,"; total estimé :",round(total,2))
    print(pred.sort_values("prevision",ascending=False)[["station_id","type_quartier","prevision","part_%"]].head(10).round(3))
    pred[["station_id","type_quartier","prevision","part_%"]].to_csv(OUT/"q3_prevision_demain_8h.csv",index=False)
    print("Limite : sans dates calendaires, ce n'est pas une vraie prévision temporelle de demain.")

    # Q4 — Test en retirant successivement un jour de semaine
    print("\\n===== Q4 =====")
    resultats=[]
    for j in JOURS:
        tr=tab[tab["jour"]!=j]; te=tab[tab["jour"]==j]
        m=RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=0)
        m.fit(tr[["heure","quartier_num"]],tr["nb_trajets"])
        e=mean_absolute_error(te["nb_trajets"],m.predict(te[["heure","quartier_num"]]))
        eb=mean_absolute_error(te["nb_trajets"],np.full(len(te),tr["nb_trajets"].mean()))
        resultats.append([j,e,eb])
    q4=pd.DataFrame(resultats,columns=["jour_teste","erreur_modele","erreur_moyenne"])
    print(q4.round(3))
    print("Ce test porte sur les jours de semaine, pas sur les jours fériés ou événements.")

    # Q5 — Non-renouvellement des annuels
    print("\\n===== Q5 =====")
    annuels=abonnes[abonnes["formule"]=="annuel"].copy()
    annuels["depart"]=1-annuels["renouvele"]
    print("Taux de non-renouvellement :",round(annuels["depart"].mean(),3))
    annuels["tranche_anciennete"]=pd.cut(annuels["anciennete_mois"],[0,6,12,24,36,48,60])
    print(annuels.groupby("tranche_anciennete",observed=False)["depart"].agg(["count","mean"]).round(2))
    annuels["quartile_trajets"]=pd.qcut(annuels["trajets_par_mois"],4,duplicates="drop")
    print(annuels.groupby("quartile_trajets",observed=False)["depart"].mean().round(2))
    print(annuels.groupby("incidents_signales")["depart"].agg(["count","mean"]).round(2))
    vars_ab=["anciennete_mois","trajets_par_mois","distance_moy_km","incidents_signales"]
    Xa=annuels[vars_ab].replace([np.inf,-np.inf],np.nan).fillna(annuels[vars_ab].median())
    ya=annuels["depart"]
    if ya.nunique()==2 and ya.value_counts().min()>=2:
        a,b,c,d=train_test_split(Xa,ya,test_size=.3,random_state=0,stratify=ya)
        sc=StandardScaler().fit(a); lg=LogisticRegression(max_iter=1000).fit(sc.transform(a),c)
        print("AUC :",round(roc_auc_score(d,lg.predict_proba(sc.transform(b))[:,1]),2))
    else: print("Régression logistique non calculée : effectif/classe insuffisant.")
    print("L'ancienneté est descriptive ; aucune date exacte de départ n'est disponible.")

    # Q6 — Simulation de campagne
    print("\\n===== Q6 =====")
    valeur_abonne=120; taux_succes=.4  # hypothèses, à confirmer
    tous=abonnes.copy(); tous["depart"]=1-tous["renouvele"]
    tous["mensuel"]=(tous["formule"]=="mensuel").astype(int)
    vars6=vars_ab+["mensuel"]
    X=tous[vars6].replace([np.inf,-np.inf],np.nan).fillna(tous[vars6].median())
    y=tous["depart"]
    if y.nunique()==2 and y.value_counts().min()>=5:
        prob=cross_val_predict(LogisticRegression(max_iter=1000),StandardScaler().fit_transform(X),y,cv=5,method="predict_proba")[:,1]
        ordre=np.argsort(-prob); cumul=np.cumsum(y.to_numpy()[ordre]); total=int(y.sum())
        if total:
            n=int(np.argmax(cumul>=.5*total)+1)
            print("Abonnés :",len(tous),"; départs :",total,"; contacts pour repérer 50% :",n)
            for cout in [2,5,10,15,20,30]:
                gains=taux_succes*valeur_abonne*cumul-cout*np.arange(1,len(cumul)+1)
                meilleur=int(np.argmax(gains)+1)
                print("Coût/contact",cout,"€ : contacts",meilleur,"; gain net simulé",round(gains.max(),2),"€")
            print("Hypothèses de simulation : valeur abonné = 120 €, succès = 40 %.")
        else: print("Aucun non-renouvellement observé.")
    else: print("Q6 non calculée : classes ou effectifs insuffisants.")

    # Q7 — Typologie des stations
    print("\\n===== Q7 =====")
    stations["trafic"]=stations["departs"]+stations["arrivees"]
    v=["interventions_mois","capacite","trafic"]
    Z=StandardScaler().fit_transform(stations[v].fillna(stations[v].median()))
    inerties=[]; sil=[]
    for k in range(2,min(9,len(stations))):
        km=KMeans(n_clusters=k,n_init=20,random_state=0).fit(Z)
        inerties.append(km.inertia_); sil.append(silhouette_score(Z,km.labels_))
        print("k =",k,"inertie =",round(km.inertia_,1),"silhouette =",round(sil[-1],2))
    k_final=4 if len(stations)>=4 else 2
    km=KMeans(n_clusters=k_final,n_init=50,random_state=0).fit(Z)
    stations["type_station"]=km.labels_
    print(stations.groupby("type_station").agg(nb_stations=("station_id","count"),interventions=("interventions_mois","mean"),capacite=("capacite","mean"),trafic=("trafic","mean")).round(1))
    print(pd.crosstab(stations["type_station"],stations["type_quartier"]))
    stations.to_csv(DATA/"stations_avec_types.csv",index=False)

    # Q8 — Rééquilibrage par type (solde sur toute la période)
    print("\\n===== Q8 =====")
    mapping=stations.set_index("station_id")["type_station"]
    trajets["type_depart"]=trajets["station_depart"].map(mapping)
    trajets["type_arrivee"]=trajets["station_arrivee"].map(mapping)
    dep=trajets["type_depart"].value_counts().sort_index()
    arr=trajets["type_arrivee"].value_counts().sort_index()
    solde=pd.DataFrame({"departs":dep,"arrivees":arr}).fillna(0)
    solde["solde"]=solde["departs"]-solde["arrivees"]
    print(solde)
    print("Pas de solde horaire fiable : l'heure réelle d'arrivée n'est pas fournie.")

    # Q9 — Carte et zones candidates
    print("\\n===== Q9 =====")
    plt.figure(figsize=(7,6))
    plt.scatter(stations["longitude_km"],stations["latitude_km"],c=stations["solde"],cmap="RdBu_r",s=60)
    plt.colorbar(label="Départs - arrivées"); plt.title("Carte des stations"); plt.xlabel("x (km)"); plt.ylabel("y (km)")
    plt.tight_layout(); plt.savefig(OUT/"q9_carte_departs_arrivees.png"); plt.close()
    xmin,xmax=stations["longitude_km"].min(),stations["longitude_km"].max()
    ymin,ymax=stations["latitude_km"].min(),stations["latitude_km"].max()
    xb=np.arange(np.floor(xmin),np.ceil(xmax)+1,1); yb=np.arange(np.floor(ymin),np.ceil(ymax)+1,1)
    if len(xb)<2: xb=np.array([xmin-.5,xmax+.5])
    if len(yb)<2: yb=np.array([ymin-.5,ymax+.5])
    nst,_,_=np.histogram2d(stations["longitude_km"],stations["latitude_km"],bins=[xb,yb])
    co=stations.set_index("station_id")[["longitude_km","latitude_km"]]
    x1=trajets["station_depart"].map(co["longitude_km"]).to_numpy()
    y1=trajets["station_depart"].map(co["latitude_km"]).to_numpy()
    x2=trajets["station_arrivee"].map(co["longitude_km"]).to_numpy()
    y2=trajets["station_arrivee"].map(co["latitude_km"]).to_numpy()
    midx=(x1+x2)/2; midy=(y1+y2)/2
    passages,_,_=np.histogram2d(midx,midy,bins=[xb,yb])
    candidats=[]
    for i in range(len(xb)-1):
        for j in range(len(yb)-1):
            if nst[i,j]==0 and passages[i,j]>=10:
                candidats.append([(xb[i]+xb[i+1])/2,(yb[j]+yb[j+1])/2,int(passages[i,j])])
    candidats=pd.DataFrame(candidats,columns=["x_km","y_km","nb_passages"])
    if len(candidats): candidats=candidats.sort_values("nb_passages",ascending=False)
    print("Zones candidates exploratoires :\\n",candidats.head(8))
    stations["trajets_par_place"]=np.where(stations["capacite"]>0,stations["trafic"]/stations["capacite"],np.nan)
    print("Stations les plus sollicitées par place :")
    print(stations.sort_values("trajets_par_place",ascending=False)[["station_id","type_quartier","capacite","trafic","trajets_par_place"]].head(8).round(2))
    candidats.to_csv(OUT/"q9_stations_a_ouvrir.csv",index=False)
    print("\\nFin des analyses Q1 à Q9. Résultats dans",OUT)

if __name__ == "__main__":
    main()
