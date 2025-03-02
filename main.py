import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("insurance.csv")

#Comprendre les données
print(data.head(30))
print(data.info())
print(data.describe())
print(data['age'].value_counts())
print(data['sex'].value_counts())
print(data['smoker'].value_counts())

#Visualiser les valeurs aberrantes
# sns.boxplot(x=data['age'])
# #plt.hist(data['age'])
# #plt.xlabel('age')
# #plt.ylabel('frequences')
# plt.title('distribution d\'age')
# plt.show()
#
# sns.boxplot(x=data['bmi'])
# #plt.hist(data['bmi'])
# #plt.xlabel('bmi')
# #plt.ylabel('frequency')
# plt.title('distribution du bmi')
# plt.show()
#
#
# sns.boxplot(x=data['charges'])
# #plt.hist(data['charges'])
# #plt.xlabel('charges')
# #plt.ylabel('frequency')
# plt.title('distribution des charges')
# plt.show()


#Detection des valeurs aberrantes
#utilisation de l'IQR : écart interquatile
def detect_val_aber(df,colonne):
    Q1 = df[colonne].quantile(0.25)
    Q3 = df[colonne].quantile(0.75)
    IQR = Q3 - Q1

    #on définit les limites
    seuil_inf = Q1 - 1.5*IQR
    seuil_sup = Q3 - 1.5*IQR

    val_aberrantes = data[(data[colonne]<seuil_inf) | (data[colonne]>seuil_sup)]
    return val_aberrantes

#appliquons la fonction sur les variables numériques
colonnes_num = ["age","bmi","charges"]
for col in colonnes_num:
    outliers = detect_val_aber(data,col)
    print(f"valeurs aberrantes pour {col}:{outliers.shape[0]} observations")
#visualiser les valeurs aberrantes dand la boite à moustache
plt.figure(figsize=(12, 5))
for i, col in enumerate(colonnes_num):
    plt.subplot(1, 3, i + 1)
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot de {col}")
plt.tight_layout()
plt.show()

#charges a des valeurs très élevées qui faussaient la détection des outliers.
#En appliquant log(charges), on réduit l’écart entre les valeurs extrêmes et la médiane,
# rendant les données plus symétriques et faciles à analyser.
data["log_charges"] = np.log(data["charges"])
# sns.boxplot(y=data["log_charges"])
# plt.title("Boxplot de log(charges)")
# plt.show()


#on a des valeurs aberrantes dans bmi et charges, on doit les standariser
#mettre toutes les variables sur la meme echelle (moyenne=0, ecart-type=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[["age","bmi","log_charges"]]=scaler.fit_transform(data[["age","bmi","log_charges"]])
print(data.head())
print(data.describe())
#les données sont bien standarisées

#passons à l'analyse des relations entre log_charges et les autres variables

#on doit en premier lieu convertir les colonnes objet en nuùérique
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["sex"]=encoder.fit_transform(data["sex"])
# 0 pour male et 1 pour female
data["smoker"]=encoder.fit_transform(data["smoker"])
#0 pour non fumeur et 1 pour fumeurs

data = pd.get_dummies(data, columns=["region"], drop_first=True)
#drop_first=true pour éviter la multicolinéarité

plt.figure(figsize=(8,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de correlation")
plt.show()
#on remarque que les variable qui sont fortement corrélé avec log_charges sont age et smoker
#ce qui n'ont presque pas d'influence sont sex et region

sns.scatterplot(x=data["bmi"], y=data["log_charges"], hue=data["smoker"])
plt.title("Relation entre BMI et Charges médicales")
plt.show()
#la relation entre bmi et charges n'est pas linéaire, pas de direct influence
#les fumeurs ont des couts médicaxux tres eleves que les non fumeurs

sns.boxplot(x=data["smoker"],y=data["log_charges"])
plt.title("Impact du tabagisme sur les charges médicaux")
plt.show()
#on remarque un écart significatif entre les fumeurs et non fumeurs en termes des couts
#difference clairement visible sur la boite a moustache(boxplot)

sns.scatterplot(x=data["age"], y=data["log_charges"])
plt.title("relation entre age et charges medicales")
plt.show()
#pour l'age on a une relation modérée mais significative, puisque la plupart des points ont une relation linéaire avec les charges


#regression linéaire
from sklearn.linear_model import LinearRegression

X=data[["age","bmi","smoker"]]
Y=data["log_charges"]

model = LinearRegression()
model.fit(X,Y)

#prediction
y_pred = model.predict(X)

#tracage de la courbe
plt.scatter(Y, y_pred, color="blue", alpha=0.5)
plt.xlabel("Valeurs réelles (log_charges)")
plt.ylabel("Valeurs prédites")
plt.title("Prédictions vs Valeurs réelles")
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

#calcul des erreurs
mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y,y_pred)

print(f"MSE = {mse}")
print(f"RMSE = {rmse}")
print(f"R2 = {r2}")

#analyse des résidus
#les residus sont la difference entre les valeurs réelles et les valeurs prédites
#un bon modele doit avoir des résidus proches de 0

residus = Y - y_pred

#verifier si les residus sont bien répartis
plt.figure(figsize=(8,6))
plt.scatter(y_pred,residus,color="blue", alpha=0.5)
plt.axhline(y=0, color="red")
plt.xlabel("valeurs predites")
plt.ylabel("residus")
plt.title("graphe des residus")
plt.show()

#on observe que la majorite des points sont répartis aléatoirement autour de 0 (ligne rouge)
#presence de quelques points lointains ce qui signifie la présence des valeurs aberrantes

sns.histplot(residus, bins=30, kde=True)
plt.xlabel("Résidus")
plt.title("Distribution des résidus")
plt.show()

#ressemeble generalement à une gaussienne avec un peu d'asymetrie
#charges medicale suivent une distribution asymetrique