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

### Question 3 : Combien de trajets partiront de chaque station demain à 8 h ?

Nous avons pris « demain » comme un jour cible à choisir. Pour illustrer la prévision, nous avons retenu un mardi et un samedi. Le fichier des trajets indique le jour de la semaine et l’heure de départ. Comme il ne précise ni les dates ni le nombre de semaines couvertes, nos estimations restent à l’échelle de l’échantillon.

#### Préparation des données et choix du modèle

Dans les données nettoyées, nous avons compté **61 départs à 8 h**, répartis sur **54 des 140 stations**. Nous avons vérifié que les identifiants des trajets étaient uniques et que chaque départ de 8 h correspondait à une station du réseau. Pour compter un départ, nous utilisons son jour, son heure et sa station de départ. Une distance ou une durée manquante ne change donc pas le comptage.

Avec aussi peu de départs, une estimation par station et par jour s’appuierait souvent sur zéro ou un seul trajet. Nous avons donc commencé par estimer le total du réseau avec une **régression linéaire**, puis nous avons réparti ce total entre les stations.

La régression prend une seule variable, `weekend`, qui vaut 0 du lundi au vendredi et 1 le samedi et le dimanche. L’ajustement donne `départs estimés = 10 − 4,5 × weekend`. Avec cette seule variable binaire, la régression revient à calculer la moyenne de chaque groupe. Elle donne **10 départs en semaine** et **5,5 le week-end**, soit un volume estimé 45 % plus bas le week-end dans l’échantillon. Un kNN sur sept totaux et une seule variable binaire dépendrait du choix de `k`, avec plusieurs jours à la même distance. K-means et DBSCAN auraient servi à trouver des groupes. Ici, nous cherchions à estimer un nombre de départs.

Pour passer du réseau aux stations, nous multiplions le total estimé par la part de chaque station dans les 61 départs observés à 8 h. Par exemple, ST212 compte deux départs sur 61. Nous lui attribuons donc `10 × 2 / 61`, soit **0,33 départ** le mardi, et `5,5 × 2 / 61`, soit **0,18 départ** le samedi. Nous gardons les valeurs fractionnaires pour conserver le total du réseau. Comme les parts sont calculées tous jours confondus, le modèle conserve le même classement le mardi et le samedi.

#### Résultats du mardi et du samedi

| Jour cible | Départs observés à 8 h | Total estimé à 8 h | Résultats par station |
| --- | ---: | ---: | --- |
| Mardi | 11 | 10 | [CSV du mardi](resultats_q3_mardi.csv) |
| Samedi | 4 | 5,5 | [CSV du samedi](resultats_q3_samedi.csv) |

La somme des 140 lignes de chaque CSV retrouve bien le total estimé. Le modèle sous-estime le mardi d’un départ et surestime le samedi de 1,5 départ. Le mardi et le samedi entrent tous deux dans le calcul des moyennes de leur groupe. Ces écarts montrent l’effet du regroupement sur ces deux exemples.

Sept stations, dont ST212, ST236 et ST239, arrivent en tête à égalité avec deux départs observés chacune à 8 h, tous jours confondus. Les 47 stations avec un seul départ observé obtiennent environ 0,16 départ le mardi et 0,09 le samedi. Les 86 autres ont une estimation nulle parce qu’aucun départ à 8 h n’y figure dans l’échantillon. Nous utiliserions donc le classement pour choisir les sept premières stations à contrôler avant 8 h. Pour fixer un nombre de vélos ou de camionnettes, il faudrait aussi tenir compte du stock de vélos disponible dans chaque station et des arrivées.

#### Vérification du modèle

Pour avoir une idée de l’erreur sur le réseau, nous avons retiré chaque catégorie de jour à tour de rôle, recalculé le modèle sur les six autres et comparé sa prévision au total laissé de côté. Nous avons fait le même calcul avec une moyenne globale comme référence. L’erreur absolue moyenne est de **2,64 départs** pour la régression et de **2,95** pour la moyenne globale. Le gain est faible sur sept catégories. Cette vérification porte sur le total réseau. La Q4 permettra de regarder de plus près les erreurs sur les catégories particulières observables dans les fichiers, comme le samedi et le dimanche.

Pour reproduire les deux CSV depuis la racine du dépôt, avec Python et pandas, utiliser PowerShell.

```powershell
python src/q3_prevision_departs.py mar | Set-Content -Encoding utf8 resultats_q3_mardi.csv
python src/q3_prevision_departs.py sam | Set-Content -Encoding utf8 resultats_q3_samedi.csv
```

L’option `--evaluation` affiche aussi les erreurs sur le total réseau.
