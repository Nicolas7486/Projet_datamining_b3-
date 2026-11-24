import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
DOSSIER = "Sujet_4_Citycycle/"          
stations = pd.read_csv(DOSSIER + "Citycycle_stations.csv")
trajets = pd.read_csv(DOSSIER + "Citycycle_trajets.csv")
print(stations.shape, trajets.shape)
trajets = trajets.drop_duplicates() 
trajets = trajets.drop_duplicates() 
trajets = trajets[trajets["duree_min"] < 1440]
print("trajets propres :", len(trajets))
