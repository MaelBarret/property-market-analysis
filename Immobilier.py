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
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, r2_score

dirname = os.path.dirname(os.path.abspath(__file__))
DEPARTEMENT = "34"
CITY = "Montpellier"
TYPE = "Maison"
MUTATION = "Vente"
MAX_PRICE = 1500000
MAX_SURFACE = 250

def download(annee, department = DEPARTEMENT):
    """
    Download rents data from https://files.data.gouv.fr concerning specified french department
    """
    CSV_URL = "https://files.data.gouv.fr/geo-dvf/latest/csv/" + annee + "/departements/"+ department + ".csv.gz"
    filename = dirname + '/data/'+ department +'_' + annee +'.csv'
    if not exists(dirname + '/data/'+ department +'_' + annee + '.csv'):
        with urllib.request.urlopen(CSV_URL) as response:
             with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()
        with open(filename, 'wb') as f:
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
    years[str(year)] = dirname + '\\data\\'+ CITY +'_' + str(year) + '.csv'

for year in years:
    download(year)

# Merge files in a single dataframe
path = dirname + '\\data\\'
all_files = glob.glob(os.path.join(path , "*.csv"))

df = pd.concat((pd.read_csv(f, sep=",", decimal='.', 
                            usecols = [0,1,3,4,10,11,30,31,32,34,37])
                             for f in all_files), ignore_index=True)

# Filtering
df = df[(df.type_local == TYPE) & (df.nom_commune == CITY) &
         (df.nature_mutation == MUTATION) &
          (df.valeur_fonciere < MAX_PRICE) & (df.surface_reelle_bati < MAX_SURFACE)]

# Remove duplicates basing on id
df = df.drop_duplicates(subset=['id_mutation'], keep='first')
df = df.dropna()
df = df.reset_index(drop = True)
X = df[['nombre_pieces_principales','surface_terrain','surface_reelle_bati']]
y = df[['valeur_fonciere']]

# Spliting test/training dataset
X_train, X_test, y_train, y_test = \
	model_selection.train_test_split(X,
                                    y,
                                	test_size=0.3 # 30% des données dans le jeu de test
                                	)

# Linear regression
lr = linear_model.LinearRegression()
lr_baseline = lr.fit(X_train[['surface_reelle_bati']], y_train)
baseline_pred = lr_baseline.predict(X_test[['surface_reelle_bati']])

def sumsq(x,y):
    return sum((x - y)**2)

def r2score(pred, target):
    return 1 - sumsq(pred, target) / sumsq(target, np.mean(target))

score_bl = r2score(baseline_pred[:,0], y_test['valeur_fonciere'])

plt.plot(X_test[["surface_reelle_bati"]], y_test, 'ro', markersize = 5)
plt.plot(X_test[["surface_reelle_bati"]], baseline_pred, color="coral", linewidth = 2)
plt.title('R²: ' + str(round(score_bl,2)))
plt.show()
print(score_bl)

# # Running Evaluation Metrics
# predictions = lr.predict(X_test[['surface_reelle_bati']])
# r2 = r2_score(predictions, y_test['valeur_fonciere'])
# rmse = mean_squared_error(y_test, predictions, squared=False)
# print('The r2 is: ', r2)
# print('The rmse is: ', rmse)

# plt.scatter(predictions, y_test, color='coral')
# plt.show()
# # The coefficients
# print("Coefficients: \n", lr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
# # The coefficient of determination: 1 is perfect prediction
# print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

# # Plot outputs
# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, color="blue", linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()

df.to_csv('34.csv')