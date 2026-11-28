# CityCycle – Projet Data Mining B3

## 1. Présentation du projet

Ce projet est réalisé dans le cadre du module **Data Mining** de la formation B3 à l'ECE Paris.

Il consiste à analyser les données d'un service de vélos en libre-service, CityCycle, afin de mieux comprendre les habitudes d'utilisation, l'activité des stations, les besoins de maintenance et les possibilités d'amélioration du service.

L'objectif est d'exploiter les données à travers différentes techniques d'analyse et de Data Mining, puis d'interpréter les résultats obtenus.

## 2. Objectifs du projet

- Explorer et comprendre les données disponibles.
- Nettoyer et préparer les jeux de données.
- Identifier les tendances d'utilisation du service.
- Étudier les facteurs associés aux déplacements.
- Réaliser des prévisions et des analyses prédictives.
- Segmenter les stations selon leurs caractéristiques.
- Identifier des pistes d'amélioration pour le service CityCycle.

## 3. Jeux de données

Le projet repose sur trois jeux de données :

| Fichier | Description |
|---|---|
| `Citycycle_trajets.csv` | Informations sur les trajets réalisés |
| `Citycycle_stations.csv` | Informations sur les stations |
| `Citycycle_abonnes.csv` | Informations sur les abonnés |

### Volume des données

- Trajets initiaux : 760 lignes.
- Trajets après suppression des doublons exacts : 700 lignes.
- Stations : 140 lignes.
- Abonnés : 500 lignes.

Les données brutes sont conservées séparément des données nettoyées et préparées.

## 4. Organisation du dépôt

```text
Projet_datamining_b3/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── nettoyagedesdonnees.py
│   ├── analyse_exploratoire.py
│   ├── preparation_donnes.py
│   └── analyse_questions.py
│
├── resultats/
│   ├── exploration/
│   ├── questions/
│   └── tableaux/
│
├── README.md
└── .gitignore
```

## 5. Méthodologie

Le projet suit plusieurs étapes :

1. **Compréhension des données :** identification des variables, des dimensions et de la qualité des jeux de données.
2. **Nettoyage :** suppression des doublons, traitement des valeurs manquantes et correction des valeurs incohérentes.
3. **Analyse exploratoire :** étude des distributions, des tendances et des relations entre les variables.
4. **Préparation des données :** création de nouvelles variables utiles aux analyses.
5. **Data Mining :** application de méthodes statistiques, prédictives et de clustering.
6. **Évaluation :** comparaison des résultats et interprétation des modèles.
7. **Visualisation :** représentation graphique des principaux résultats.

## 6. Questions étudiées et résultats

### Q1 – Interventions, activité des stations et heures de pointe

L'analyse met en évidence une moyenne de 11,48 interventions mensuelles dans les quartiers de gare, contre environ 4 interventions dans les autres types de quartiers.

La corrélation entre les interventions et les départs est faible (-0,13). Les données montrent également des différences entre les départs et les arrivées selon les quartiers.

Les heures de départ les plus actives se situent notamment entre 7 h et 9 h et entre 15 h et 18 h.

### Q2 – Facteurs associés à l'utilisation du service

Le modèle indique les importances suivantes pour les variables étudiées :

- Heure : 55 %.
- Quartier : 26 %.
- Jour : 19 %.

La vitesse moyenne observée est de 10,03 km/h sans pluie et de 8,90 km/h avec pluie.

Ces résultats décrivent les trajets réalisés et ne permettent pas, à eux seuls, de mesurer la demande non satisfaite.

### Q3 – Prévision de l'activité

Un modèle de prévision a été évalué et comparé à une référence basée sur la moyenne.

- MAE du modèle : 0,0708.
- MAE de la moyenne : 0,0714.
- Estimation produite pour le dimanche 29 novembre 2026 : 6,16.

Le modèle présente une erreur légèrement inférieure à celle de la référence.

Cette estimation reste indicative, car les données ne contiennent pas de dates calendaires complètes.

### Q4 – Évaluation du modèle selon les jours

Une évaluation par retrait successif de chaque jour de la semaine a été réalisée.

L'erreur du modèle est inférieure à celle de la référence basée sur la moyenne pour les sept jours testés.

Ce test permet d'évaluer le comportement du modèle selon les jours de la semaine, mais ne constitue pas une évaluation sur les jours fériés ou les événements particuliers.

### Q5 – Analyse du renouvellement des abonnements

Le taux de non-renouvellement observé est de 21,7 %.

L'analyse étudie notamment :
- L'ancienneté des abonnés.
- Le nombre de trajets réalisés.
- Les incidents signalés.

Le modèle de classification obtient une AUC de 0,57, ce qui indique une capacité de discrimination limitée.

### Q6 – Simulation de rentabilité d'une campagne de fidélisation

Une simulation a été réalisée pour étudier différents coûts de contact des abonnés.

Les hypothèses retenues sont :
- Valeur estimée d'un abonné : 120 €.
- Taux de réussite de la campagne : 40 %.

Les simulations montrent un gain net positif dans les scénarios testés.

Ces résultats sont théoriques et dépendent directement des hypothèses retenues. Ils ne représentent pas une rentabilité réellement observée.

### Q7 – Segmentation des stations par clustering

Une méthode de clustering K-Means a été appliquée pour regrouper les stations selon leurs caractéristiques.

Plusieurs nombres de clusters ont été testés, de K = 2 à K = 8.

Le meilleur score de silhouette obtenu est de 0,29 pour K = 4.

Quatre groupes de stations ont ainsi été étudiés selon le nombre de stations, les interventions, la capacité et le trafic.

La séparation entre les groupes reste modérée.

### Q8 – Analyse des déséquilibres entre départs et arrivées

L'analyse compare les volumes de départs et d'arrivées dans les quatre groupes de stations.

Les soldes obtenus sont :

- Groupe 1 : -25.
- Groupe 2 : -19.
- Groupe 3 : +15.
- Groupe 4 : +29.

Les groupes présentant un solde négatif ont davantage de départs que d'arrivées, tandis que les groupes présentant un solde positif ont davantage d'arrivées.

L'heure réelle d'arrivée n'étant pas disponible, aucune analyse horaire fiable du rééquilibrage n'a été réalisée.

### Q9 – Identification de zones candidates pour de nouvelles stations

Une analyse spatiale exploratoire a permis d'identifier six zones candidates à partir de la répartition des passages.

Les coordonnées des zones identifiées sont notamment :

- (6,5 ; 4,5) : 69 passages.
- (4,5 ; 4,5) : 37 passages.
- (4,5 ; 3,5) : 24 passages.
- (3,5 ; 3,5) : 22 passages.
- (7,5 ; 4,5) : 22 passages.
- (6,5 ; 2,5) : 11 passages.

L'analyse identifie également les stations présentant un nombre élevé de trajets par place disponible.

Ces zones constituent des pistes d'implantation à approfondir et doivent être validées à l'aide d'informations géographiques et opérationnelles complémentaires.

## 7. Technologies utilisées

- **Python** : traitement et analyse des données.
- **Pandas** : manipulation des jeux de données.
- **NumPy** : calcul numérique.
- **Matplotlib / Seaborn** : visualisation des résultats.
- **Scikit-learn** : apprentissage automatique et clustering.
- **Git / GitHub** : gestion de versions et collaboration.

## 8. Exécution du projet

Depuis la racine du dépôt, les scripts peuvent être exécutés dans l'ordre suivant :

```bash
python src/nettoyagedesdonnees.py
python src/analyse_exploratoire.py
python src/preparation_donnes.py
python src/analyse_questions.py
```

Les résultats graphiques et les tableaux sont enregistrés dans le dossier `resultats/`.

## 9. Limites de l'étude

Plusieurs limites doivent être prises en compte dans l'interprétation des résultats :

- L'absence de dates calendaires complètes limite les possibilités de prévision temporelle.
- Les heures disponibles correspondent aux heures de départ et non aux heures réelles d'arrivée.
- Les données décrivent uniquement les trajets réalisés.
- Le modèle de renouvellement présente une performance limitée.
- La simulation financière repose sur des hypothèses.
- Les zones candidates à l'implantation nécessitent une validation complémentaire.

## 10. Conclusion

Ce projet a permis de mettre en œuvre différentes étapes d'un processus de Data Mining, depuis le nettoyage des données jusqu'à l'application de méthodes prédictives et de clustering.

Les analyses ont permis de mieux comprendre l'activité des stations, les habitudes de déplacement, les facteurs associés à l'utilisation du service et les possibilités d'amélioration du réseau CityCycle.

Les résultats obtenus constituent une base d'aide à la décision, tout en tenant compte des limites des données et des modèles utilisés.
