
import sys
import pandas as pd
import numpy as np

from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense

# print(sys.version)


data = pd.read_table("datas/data_1.txt",sep="\t",header=0)

print(data.info())

data['Y'] = (data['Y'] == 'pos').astype(int)
# X = [data.iloc[:,:2]]
# y=[data]
# XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,y,test_size=500,random_state=1,stratify=y)
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data[['X1', 'X2']].values, data['Y'].values, test_size=0.5)

#Init sequential neurone

modelsimple = Sequential()

# le model une couche
#tf_neurones_entrees_x = tf.placeholder(tf.float32,[None,2])
modelsimple.add(Dense(units=1,input_dim=2,activation="sigmoid"))

print(modelsimple.get_config())

#compilation de l'apprentissage
modelsimple.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

#l'apprentissage
modelsimple.fit(X_train,Y_train,epochs=150,batch_size=10)

print(modelsimple.get_weights())

#prediction sur X_test
predsimple = modelsimple.predict(X_test)
