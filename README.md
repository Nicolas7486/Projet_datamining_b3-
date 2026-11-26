City Cycle – Projet Data Mining B3

1. Présentation du projet

Ce projet consiste à analyser les données du service de vélos en libre-service City Cycle dans le cadre du module Data Mining.

2. Objectifs

Explorer et comprendre les données.
Nettoyer et préparer les jeux de données.
Réaliser des analyses et appliquer des méthodes de Data Mining.
Interpréter les résultats obtenus.

3. Jeux de données

Le projet utilise trois fichiers :

Citycycle_stations.csv
Citycycle_trajets.csv
Citycycle_abonnes.csv

4. Organisation du dépôt

Projet_datamining_b3/

data/
raw/
processed/

notebooks/

src/

README.md

.gitignore

nettoyagedesdonnees.py

5. Méthodologie

Compréhension des données
Nettoyage des données
Analyse exploratoire
Application des méthodes de Data Mining
Évaluation et interprétation des résultats

6. Résultats

Question 1 : stations à problème et heures de pointe

On a d'abord nettoyé les trajets : doublons, distances nulles ou négatives, et durées de 1440 minutes ou plus (une journée entière, ce qui n'est pas réaliste). Il reste donc 692 trajets sur 760 

au niveau de stations, les gares sortent clairement du lot : 11,48 interventions par mois en moyenne, contre environ 4 dans les autres quartiers (résidentiel 3,85, bureaux 3,90, université 4,21). 14 stations ont donc au moins 10 interventions par mois, et ce sont toutes des gares. Les 20 % de stations les plus touchées (28 stations) concentrent 41 % des interventions.

Pour les heures de pointe, on observe deux pics : le premier c'est un le matin vers 8 h (61 départs) et un en fin d'après-midi avec le maximum à 16 h (73 départs). La nuit par contre , il y a très peu de départs (moins de 10 à 5 h, 22 h et 23 h). Ça ressemble à des trajets domicile-travail.

En conclusion, la maintenance devrait se concentrer sur les gares, et éviter les heures de pointe (7 h à 9 h et 15 h à 18 h) pour ne pas gêner les usagers.
