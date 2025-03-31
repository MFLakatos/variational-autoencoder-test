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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

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
important_elements.append("Fraude clase 1")
important_elements.append("Normal clase 0")

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
# Guardar la figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
f.savefig('imgs/timeVsAmount.png', bbox_inches='tight')
f.savefig('imgs/timeVsAmount.pdf', bbox_inches='tight')
#-----------------------------------------------------------------------------------------------------------------------
#------------- Prepare data
# Eliminar la columna Time por no ser relevante
# Estandarizar datos
# Dividir en train y test. Train: solo clases correctas
#-----------------------------------------------------------------------------------------------------------------------

data: pd.DataFrame = df.drop(['Time'], axis=1)
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

X_train: pd.DataFrame
X_test: pd.DataFrame
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Class == 0]
X_train = X_train.drop(['Class'], axis=1)
y_test: pd.Series = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)

# Pandas -> Numpy
X_train: np.ndarray = X_train.values
X_test: np.ndarray = X_test.values
print(X_train.shape)

#-----------------------------------------------------------------------------------------------------------------------
#------------- Build the model
#-----------------------------------------------------------------------------------------------------------------------


input_dim = X_train.shape[1]
encoding_dim = 14
important_elements.append(f"input dim: {input_dim}")
important_elements.append(f"encoding dim: {encoding_dim}")

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)


#-----------------------------------------------------------------------------------------------------------------------
#------------- Train
#-----------------------------------------------------------------------------------------------------------------------
nb_epoch = 100
batch_size = 32
important_elements.append(f"number epoch: {nb_epoch}")
important_elements.append(f"batch size: {batch_size}")
optimizer = 'adam'
loss='mean_squared_error',
metric= 'accuracy'
important_elements.append(f"optimizer: {optimizer}")
important_elements.append(f"loss: {loss}")
important_elements.append(f"metric: {metric}")

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='/media/old-tf-hackers-7/logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
autoencoder = load_model('model.h5')

#-----------------------------------------------------------------------------------------------------------------------
#------------- Evaluation
#-----------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
# Guardar la segunda figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
plt.savefig('imgs/modelLoss.png', bbox_inches='tight')
plt.savefig('imgs/modelLoss.pdf', bbox_inches='tight')

#-----------------------------------------------------------------------------------------------------------------------
#------------- Error distribution
# MSE
# Distribución MSE sin fraudes
# Distribución MSE de fraudes
# Recall, precision
#-----------------------------------------------------------------------------------------------------------------------
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
important_elements.append(f"MSE: {mse}")
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()

# Reconstrucción del error SIN Fraude
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)

# Reconstrucción del error CON Fraude
fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = error_df[error_df['true_class'] == 1]
_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=10)

# Curva ROC
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# Guardar la segunda figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
plt.savefig('imgs/ROC.png', bbox_inches='tight')
plt.savefig('imgs/ROC.pdf', bbox_inches='tight')

# precision, recall 
precision, recall, th = precision_recall_curve(error_df.true_class, error_df.reconstruction_error)
plt.figure()
plt.plot(recall, precision, 'b', label='Precision-Recall curve')
plt.title('Recall vs Precision')
plt.xlabel('Recall')
plt.ylabel('Precision')
# Guardar la segunda figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
plt.savefig('imgs/recallVsPrecision.png', bbox_inches='tight')
plt.savefig('imgs/recallVsPrecision.pdf', bbox_inches='tight')

#-----------------------------------------------------------------------------------------------------------------------
#------------- Prediction
#-----------------------------------------------------------------------------------------------------------------------
threshold = 2.9
important_elements.append(f"threshold: {threshold}")
groups = error_df.groupby('true_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
# Guardar la segunda figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
plt.savefig('imgs/reconstructionError.png', bbox_inches='tight')
plt.savefig('imgs/reconstructionError.pdf', bbox_inches='tight')

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
conf_matrix = confusion_matrix(error_df.true_class, y_pred)
plt.figure()
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
# Guardar la segunda figura en formato PNG y PDF en la carpeta 'imgs' con formato tight
plt.savefig('imgs/ConfusionMatrixTh'+str(threshold) +'.png', bbox_inches='tight')
plt.savefig('imgs/ConfusionMatrixTh'+str(threshold) +'.pdf', bbox_inches='tight')

#-----------------------------------------------------------------------------------------------------------------------
#------------- Convertir e imprimir tiempo de ejecución
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
#------------- Save data + plt.show()
#-----------------------------------------------------------------------------------------------------------------------
# Información del autoencoder
autoencoder_info = (
    "Our Autoencoder uses 4 fully connected layers with 14, 7, 7 and 29 neurons respectively. "
    "The first two layers are used for our encoder, the last two go for the decoder. "
    "Additionally, L1 regularization will be used during training."
)

# Guardamos la información del modelo y los elementos importantes en un archivo de texto
with open('infoFileOutput.txt', 'w') as f:
    # f.write(f"Modelo utilizado: {model}\n")
    f.write("Información del Autoencoder:\n")
    f.write(autoencoder_info + "\n")
    f.write("Compilación del Autoencoder:\n")
    f.write("Optimizer: adam\n")
    f.write("Loss: mean_squared_error\n")
    f.write("Metrics: accuracy\n")
    f.write("Callbacks:\n")
    f.write(" - ModelCheckpoint: model.h5, save_best_only=True\n")
    f.write(" - TensorBoard: log_dir='/media/old-tf-hackers-7/logs'\n")
    if important_elements:  # Verificamos si la lista no está vacía
        f.write("Elementos importantes:\n")
        for element in important_elements:
            f.write(element + '\n')
    f.write(f"El tiempo de ejecución fue de: {formatted_time}")

# Mostrar las gráficas
plt.show()
