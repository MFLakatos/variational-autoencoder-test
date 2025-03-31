# https://curiousily.com/posts/credit-card-fraud-detection-using-autoencoders-in-keras/

import time
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import os
#  oneDNN (oneAPI Deep Neural Network Library)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # disable these optimizations to ensure consistent numerical results in tf
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import kagglehub
#-----
# Comienzo de conteo de tiempo de ejecución del código
#-----
# Inicia el temporizador
start_time = time.time()
#-----------------------------------------------------------------------------------------------------------------------
#------------- Configure parameters
#-----------------------------------------------------------------------------------------------------------------------

# %matplotlib inline
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]
# Lista para almacenar los elementos importantes
important_elements = []
# important_elements.append(element)

# Verificar si la carpeta 'imgs' existe, si no, crearla
if not os.path.exists('imgs'):
    os.makedirs('imgs')

#-----------------------------------------------------------------------------------------------------------------------
#------------- Import data
#-----------------------------------------------------------------------------------------------------------------------

# Descargar el dataset y obtener la ruta
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
file_name = "creditcard.csv"
file_path = os.path.join(path, file_name)

# Cargar el conjunto de datos
df = pd.read_csv(file_path)

#-----------------------------------------------------------------------------------------------------------------------
#------------- Exploration of data
#-----------------------------------------------------------------------------------------------------------------------

print(df.head())
print(f"La forma del df es: {df.shape}")
print(f"¿Hay valores nulos en el dataset? {'Sí' if df.isnull().values.any() else 'No'}")
# Separar el conjunto de datos en características y variable objetivo
X = df.drop('Class', axis=1)
y = df['Class']
print(f"Clases en la columna Class: {df['Class'].unique()}")

# count_classes = pd.Series(df['Class']).value_counts(sort=True)
# count_classes.plot(kind = 'bar', rot=0)
# plt.title("Transaction class distribution")
# plt.xticks(range(2), LABELS)
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# Suponiendo que df ya está definido y contiene el conjunto de datos
count_classes = pd.Series(df['Class']).value_counts(sort=True)
ax = count_classes.plot(kind='bar', rot=0)

# Agregar valores a la leyenda
for i, count in enumerate(count_classes):
    ax.text(i, count + 10, str(count), ha='center')

plt.title("Transaction class distribution")
plt.xticks(range(2), ['Non-Fraud', 'Fraud'])
plt.xlabel("Class")
plt.ylabel("Frequency")
# plt.legend(['Count'])

frauds = df[df.Class == 1]
normal = df[df.Class == 0]

print(f"Cantidad de fraudes: {frauds.shape[0]} y cantidad de normales: {normal.shape[0]}.")

print("Descripción de fraudes:\n",frauds.Amount.describe())

print("Descripción de normal:\n",normal.Amount.describe())

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(frauds.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
# Guardar la figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
f.savefig('imgs/fraudNormal.png', bbox_inches='tight')
f.savefig('imgs/fraudNormal.pdf', bbox_inches='tight')


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Time of transaction vs Amount by class')
ax1.scatter(frauds.Time, frauds.Amount)
ax1.set_title('Fraud')
ax2.scatter(normal.Time, normal.Amount)
ax2.set_title('Normal')
plt.xlabel('Time (in Seconds)')
plt.ylabel('Amount')
#-----------------------------------------------------------------------------------------------------------------------
#------------- Divide data in train and test
#-----------------------------------------------------------------------------------------------------------------------

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba con clases balanceadas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Combinar las características y la variable objetivo nuevamente en DataFrames
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

#-----------------------------------------------------------------------------------------------------------------------
#-------------Convertir e imprimir tiempo de ejecución + plt.show()
#-----------------------------------------------------------------------------------------------------------------------

# Finaliza el temporizador
end_time = time.time()
# Calcula el tiempo transcurrido
elapsed_time = end_time - start_time
# Convierte el tiempo transcurrido al formato HH:MM:SS
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
formatted_time = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
# Muestra el tiempo transcurrido en formato HH:MM:SS
print(f"\nFin. El tiempo de ejecución fue: {formatted_time}")
#-----------------------------------------------------------------------------------------------------------------------
#------------- Save data
#-----------------------------------------------------------------------------------------------------------------------
with open('infoFileOutput.txt', 'w') as f:
    for element in important_elements:
        f.write(element + '\n')

# Mostrar la gráfica
plt.show()
