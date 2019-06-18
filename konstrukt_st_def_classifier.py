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

#stvaramo vlastiti input_fn za slanje podataka koje koristi procjenitelj (estimator) za treniranje
#argumenti:
#1. features: znacajke koje cemo koristiti u treniranju
#2. labels: oznake koje cemo koristiti u treniranju
#3. batch_size: velicina serije (skupa) podataka koje cemo koristiti u treniranju
#funkcija vraca seriju znacajki i oznaka podataka koje cemo koristiti u treniranju
def create_training_input_fn(features, labels, batch_size, num_epochs=None, shuffle=True):
	
	def _input_fn(num_epochs=None, shuffle=True):
		
		#permutiramo podatke da dobijemo dobar uzorak
		indeks = np.random.permutation(features.index)
		#kopiramo polja znacajki i oznaka koje nam trebaju
		_features = {"pixels":features.reindex(indeks)}
		_targets = np.array(labels[indeks])
		
		#uzimamo random element i pretvorimo ga u tensor_slices jer tensorflow radi s takvim podacima
		ds = Dataset.from_tensor_slices((_features,_targets)) # 2GB limit
		#gradimo skupove podataka na kojima cemo kasnije trenirati klasifikator
		ds = ds.batch(batch_size).repeat(num_epochs)
    
		#permutiramo podatke za treniranje
		if shuffle:
			ds = ds.shuffle(10000)
    
		#vraca znacajke i oznake podataka u slijedecem skupu podataka za treniranje
		feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
		return feature_batch, label_batch
	return _input_fn




#stvaramo vlastiti input_fn za slanje podataka koje koristi procjenitelj (estimator) za predviđanje oznaka
#argumenti:
#1. features: znacajke koje cemo koristiti u predviđaju
#2. labels: oznake koje cemo koristiti u predviđanju
#3. batch_size: velicina serije (skupa) podataka koje cemo koristiti u predviđanju
#funkcija vraca seriju znacajki i oznaka podataka koje cemo koristiti upredviđanju
def create_predict_input_fn(features, labels, batch_size):
	
	def _input_fn():
		#kopiramo polja znacajki i oznaka koje nam trebaju
		_features = {"pixels": features.values}
		_targets = np.array(labels)
		
		#uzimamo random element i pretvorimo ga u tensor_slices jer tensorflow radi s takvim podacima
		ds = Dataset.from_tensor_slices((_features, _targets)) # 2GB limit
		#gradimo skup podataka kojima cemo kasnije predviđati oznake
		ds = ds.batch(batch_size)
		
		#vraca znacajke i oznake podataka u slijedecem skupu podataka za predviđanje
		feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
		return feature_batch, label_batch
	return _input_fn




#funkcija za treniranje našeg modela klasifikacije na skupu podataka za treniranje
#argumenti:
#1. learning_rate: stopa ucenja koju ce nas model koristiti,  "float"
#2. steps: ukupan broj koraka koristeci jedan batch, "int"
#3. batch_size: velicina batch-a, "int"
#4. hidden_units: specificira broj neurona u svakom sloju mreze, "lista int-ova"
#5. training_examples: "DataFrame" koji sadrzi znacajke podataka iz skupa za treniranje
#6. training_targets: "DataFrame" koji sadrzi oznake podataka iz skupa za treniranje
#7. validation_examples: "DataFrame" koji sadrzi znacajke podataka iz skupa za provjeru
#8. validation_targets: "DataFrame" koji sadrzi oznake podataka iz skupa za provjeru
#funkcija vraca istrenirani klasifikator ("DNNClassifier" objekt, dense feed-forward neural networks)
def train_nn_classification_model(learning_rate,steps,batch_size,hidden_units,training_examples,training_targets,validation_examples,validation_targets):

	per = 10
	koraci_po_per = steps / per  

	#stvori input funkcije, koriste se kako bi se izracunale vjerojatnosti pomocu kojih se racuna greska modela, postotak tocno predvidenih slova
	predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
	predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
	training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)
  
	#stvori svojstvene stupce (784 broja koji oznacavaju nijansu pojedinog piksela) koji nam koriste za predvidanje u modelu
	feature_columns = [tf.feature_column.numeric_column('pixels', shape=784)]

	_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
	
	# ako kod koraka unatrag gradijenti postanu preveliki "rezemo" ih ako postanu veci od 5.0 da ne bi dosli do NaN vrijednosti ( tj. overflow)
	_optimizer = tf.contrib.estimator.clip_gradients_by_norm(_optimizer, 5.0)
	
	#stvori klasifikator (DNNClassifier objekt)
	classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,n_classes=26,hidden_units=hidden_units,optimizer=_optimizer,config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1))
	
	# treniramo model u petlji da mozemo biljeziti greske (LogLoss error)
	# spremamo rezultate gresaka u treniranju i validaciji kako bismo mogli na kraju iscrtati graf
	training_errors = []
	validation_errors = []
	
	
	
	print("Model se trenira...")
	print("LogLoss error (na validation data):")
	for i in range (0, per):
		# treniramo model, stratamo od prijasnjeg stanja
		classifier.train(input_fn=training_input_fn,steps=koraci_po_per)
  
		# racunamo vjerojatnosti za treniranje
		training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
		training_probabilities = np.array([item['probabilities'] for item in training_predictions])
		training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
		training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,26)
        
		# racunamo vjerojatnosti za provjeru
		validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
		validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    
		validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
		validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,26)    
    
		# racunamo postotke gresaka na podacima za treniranje i provjeru te ih ispisujemo
		training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
		validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
		print("  period %02d : %0.2f" % (i, validation_log_loss))
		
		# spremamo greske u pripadne liste
		validation_errors.append(validation_log_loss)
		training_errors.append(training_log_loss)
	print("Treniranje modela zavrseno.")
	
	
	
	#oslobađamo memoriju
	_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))
  
	#izracunavamo zavrsne predikcije
	final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
	final_predictions = np.array([item['class_ids'][0] for item in final_predictions])
  
	#racunamo ukupnu tocnost na podacima za provjeru
	accuracy = metrics.accuracy_score(validation_targets, final_predictions)
	print("Final accuracy (na validation data): %0.2f" % accuracy)



	# crtamo graf gresaka koje smo dobili tijekom testiranja (LogLoss)
	plt.xlabel("Period")
	plt.ylabel("LogLoss")
	plt.plot(validation_errors, label="validacija")
	plt.plot(training_errors, label="treniranje")
	plt.title("LogLoss vs. Periods")
	plt.legend()
	plt.show()
  
	imena_oznaka=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
	
	
	
	# crtamo matricu konfuzije
	mat_konfus=metrics.confusion_matrix(validation_targets, final_predictions)
	#normalizira matricu po recima (zbroj svakog retka je 1)
	mat_konf_normalized2 = mat_konfus.astype("float") / mat_konfus.sum(axis=1)[:, np.newaxis]
	print(mat_konfus)
	plt.imshow(mat_konf_normalized2,interpolation="nearest",cmap="gist_yarg")
	plt.title("Matrica konfuzije")
	plt.colorbar()
	tick_marks2 = np.arange(len(imena_oznaka))
	plt.xticks(tick_marks2,imena_oznaka,rotation=90)
	plt.yticks(tick_marks2,imena_oznaka)
	plt.ylabel("Prava oznaka")
	plt.xlabel("Predvidena oznaka")
	plt.tight_layout()
	plt.show()
	
	#funkcija train_nn_classification_model vraća klasifikator
	return classifier