from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

#postavljamo način rada tensorflowa – želimo dobiti one poruke koje imaju label ERROR
#ostali načini su: debug, info, warn, fatal
tf.logging.set_verbosity(tf.logging.ERROR)
#postavljamo broj redaka koji se prikazuju u ispisanom prozoru
pd.options.display.max_rows = 20
#postavljamo format ispisa decimalnih brojeva na jednu decimalu
pd.options.display.float_format = '{:.1f}'.format

#učitamo skup podataka (slika) na kojima ćemo trenirati, provjeravati i testirati neuronsku mrežu
letters_orig_dataframe = pd.read_csv("A_Z_Handwritten_Data.csv",sep=",",header=None)

#permutiramo podatke jer su u prethodnoj datoteci sortirani po slovu
letters_dataframe_perm = letters_orig_dataframe.reindex(np.random.permutation(letters_orig_dataframe.index))

#odredimo broj podataka na kojima ćemo trenirati model
letters_dataframe = letters_dataframe_perm.head(10000)

#datoteka u kojoj stvaramo oznake i značajke
import oznake_feat as prvi 

#stvaramo oznake i značajke za prvih 6000 podataka, te podatke koristimo za treniranje neuronske mreže
#u train_examples spremamo svojstva na kojima cemo trenirati, a u train_targets oznake pomocu kojih provjeravamo tocnost
train_examples, train_targets = prvi.stvori_oznake_i_znacajke(letters_dataframe[:6000])

#stvaramo oznake i značajke za ostalih 4000 podataka, te podatke koristimo za provjeru (validaciju)
validation_examples, validation_targets = prvi.stvori_oznake_i_znacajke(letters_dataframe[6000:10000])

#oznake klasa su brojevi od 0 do 25, ovdje im pridružujemo slova koja predstavljaju
imena_oznaka=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']





#datoteka u kojoj definiramo model i input funkcije
import konstrukt_st_def_classifier as drugi

#treniramo model na podacima za treniranje, mozemo promijeniti parametre
#spremamo istrenirani klasifikator
classifier = drugi.train_nn_classification_model(learning_rate=0.05,steps=7500,batch_size=60,hidden_units=[200, 200, 200],training_examples=train_examples,training_targets=train_targets,validation_examples=validation_examples,validation_targets=validation_targets)
  

#stvaramo oznake i značajke podataka za testiranje točnosti našeg klasifikatora, odabrali smo 2500 podataka
test_examples, test_targets = prvi.stvori_oznake_i_znacajke(letters_dataframe_perm[70000:72500])


#pozivamo predict_input_fn funkciju
predict_test_input_fn = drugi.create_predict_input_fn(test_examples, test_targets, batch_size=100)

#spremamo predviđene oznake slova testnih podataka koje je predvidio istrenirani klasifikator
test_predictions = list(classifier.predict(input_fn=predict_test_input_fn))

#uspoređujemo predviđene oznake sa stvarnim oznakama (test_targets)
test_probabilities = np.array([item['probabilities'] for item in test_predictions])
test_pred_class_id = np.array([item['class_ids'][0] for item in test_predictions])

#računamo preciznost s kojom možemo znati da ce neko slovo biti tocno prepoznato
preciznost = metrics.accuracy_score(test_targets, test_pred_class_id)
print("%0.2f" % preciznost)




#za prvih 100 podataka iz test_podataka ispisujemo pravu oznaku, predvidenu oznaku i vjerojatnost
test_values=test_targets.values
examples_values=test_examples.values

for i in range(100):
	predict_num=test_pred_class_id[i]
	pred=imena_oznaka[predict_num]
	ozn=test_values[i]
	stv=imena_oznaka[ozn]
	prob=test_probabilities[i]
	prob_pred=prob[predict_num]
	print('prava oznaka: {0} predvidena oznaka: {1} '.format(stv,pred))
	print('vjerojatnost %0.4f' % prob_pred)




	
#uzimamo 5 slova koja dodatno i crtamo
for i in range(5):
	predict_num=test_pred_class_id[i]
	pred=imena_oznaka[predict_num]
	ozn=test_values[i]
	stv=imena_oznaka[ozn]
	prob=test_probabilities[i]
	prob_pred=prob[predict_num]
	print('prava oznaka: {0} predvidena oznaka: {1} '.format(stv,pred))
	print('vjerojatnost %0.2f' % prob_pred)
	_, ax = plt.subplots()
	ax.matshow(examples_values[i].reshape(28, 28))
	ax.set_title("Oznaka: %s predvidena %s vjer %0.4f" % (stv,pred,prob_pred))
	ax.grid(False)
	plt.show()




#crtamo matricu konfuzije
mat_konfus=metrics.confusion_matrix(test_targets, test_pred_class_id)
mat_konf_normalized2 = mat_konfus.astype("float") / mat_konfus.sum(axis=1)[:, np.newaxis]
print(mat_konfus)
plt.imshow(mat_konf_normalized2,interpolation="nearest",cmap="gist_yarg")
plt.title("Matrica konfuzije na testnim podacima")
plt.colorbar()
tick_marks2 = np.arange(len(imena_oznaka))
plt.xticks(tick_marks2,imena_oznaka,rotation=90)
plt.yticks(tick_marks2,imena_oznaka)
plt.ylabel("Prava oznaka")
plt.xlabel("Predvidena oznaka")
plt.tight_layout()
plt.show()