from os.path import exists
import os
import glob
import datetime
import pandas as pd
import numpy as np
import gzip
import urllib.request
import matplotlib.pyplot as plt
from sklearn import linear_model
import csv
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neighbors, metrics
from sklearn.metrics import mean_squared_error, r2_score

dirname = os.path.dirname(os.path.abspath(__file__))
#department = "34"

def download(annee, department = "34"):
    """
    Download rents data from https://files.data.gouv.fr concerning specified french department
    """
    CSV_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/" + annee + "/departements/"+ department + ".csv.gz"
    filename = dirname + '/data/34_' + annee +'.csv'
    if not exists(dirname + '/data/34_' + annee + '.csv'):
        with urllib.request.urlopen(CSV_URL) as response:
             with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()
        with open(filename, 'wb') as f:
            #file_content = file_content.replace(bytes(',', 'utf8'),bytes(';', 'utf8'))
            f.write(file_content)
    #if data not available -> pass 

# Download loop fetching data from the last 5 years
years = {}
for i in range(5):
    #if data not available -> décaler à l'année antérieure
    currentDateTime = datetime.datetime.now()
    date = currentDateTime.date()
    year = int(date.strftime("%Y"))
    year -= i + 1
    years[str(year)] = dirname + '\\data\\34_' + str(year) + '.csv'

for year in years:
    download(year)

# Merge files in a single dataframe
path = dirname + '\\data\\'
all_files = glob.glob(os.path.join(path , "*.csv"))

df = pd.concat((pd.read_csv(f, sep=",", decimal='.', 
                            usecols = [0,1,3,4,10,11,30,31,32,34,37])
                             for f in all_files), ignore_index=True)

# Filtering
df = df[(df.type_local == 'Maison') & (df.nom_commune == 'Montpellier') &
         (df.nature_mutation == 'Vente') & (df.valeur_fonciere == df.valeur_fonciere) &
          (df.valeur_fonciere < 10000000)]

# # Reshape data to include gardens (data aggregation)
#df['key']=df.groupby(['id_mutation','valeur_fonciere']).cumcount()

# df = df.pivot(columns=['nature_culture'], values = ['date_mutation', 'nature_mutation',
#         'valeur_fonciere', 'code_commune', 'nom_commune', 'surface_reelle_bati',
#        'nombre_pieces_principales', 'type_local', 'surface_terrain'])

# Remove duplicates based on id
df = df.drop_duplicates(subset=['id_mutation'], keep='first')
df = df.reset_index(drop = True)
x = df['surface_reelle_bati'].values
X = df.loc[:,['date_mutation','nom_commune','nombre_pieces_principales','surface_terrain','surface_reelle_bati']].values

y = df['valeur_fonciere'].values
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

# Séparation test/entrainement
X_train, X_test, y_train, y_test = \
	model_selection.train_test_split(X, y,
                                	test_size=0.3 # 30% des données dans le jeu de test
                                	)

# Visualisation
# Prix/surface
plt.scatter(X[:,4], y, color='coral')
plt.show()
# Régression linéaire
regr = linear_model.LinearRegression()
regr.fit(x, y)


# Running Evaluation Metrics
predictions = regr.predict(X_test)
r2 = r2_score(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
print('The r2 is: ', r2)
print('The rmse is: ', rmse)

plt.scatter(y, predictions, color='coral')
plt.show()
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test, color="black")
plt.plot(X_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

regr.predict(100)




# #Entraînement d'un kNN avec k = 11
# knn = neighbors.KNeighborsRegressor(n_neighbors=11)

# knn.fit(x_train, y_train)
# A = knn.kneighbors_graph(x)

# #Application pour prédire jeu de test
# y_pred = knn.predict(x_test)

# #RMSE associée
# print("RMSE : {:.2f}".format(np.sqrt( metrics.mean_squared_error(y_test, y_pred) )))

# plt.scatter(y_test, y_pred, color='coral')
# plt.show()

# y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)
# #RMSE bien plus élevé qu'avec le modèle, qui a donc bien réussi à apprendre
# print("RMSE : {:.2f}".format(np.sqrt(metrics.mean_squared_error(y_test, y_pred_random))))
# #Write dataframe to csv

# print(knn.predict(100))

df.to_csv('34.csv')

#Ajouter un ratio prix/surface permettant de réaliser une classification (donné/peu cher/cher/un peu cher/très cher)
#Distinguer bâti/jardins -> surface terrain/bâti
#+ Classes d'éloignement

#Modélisation

#Ajouter la distance à Mtp/une agglo ? -> Si Mtp seul, villes < 50 km ?