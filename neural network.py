import numpy as np
import keras
from keras.layers import Input,Dense
from keras.models import Model
from sklearn.feature_extraction.text import CountVectorizer

x=[]
y=[]
k=0
f=open('clean_ElectionGujrat2017.txt')
for i in f:
    i=i.split('|')
    x.append(i[3])
    y.append(i[len(i)-1])
    y[k]=y[k].replace('\n','')
    k+=1

bow=CountVectorizer()
x=bow.fit_transform(x)
x=x.todense()
bow=bow.vocabulary_
#print x,y
one_hot_y=keras.utils.to_categorical(y)
#x=x_train.reshape((60000,784))
inp=Input(shape=(13929,))
hid1=Dense(10,activation='sigmoid')(inp)
out=Dense(16,activation='sigmoid')(hid1)
model=Model(inputs=inp,outputs=out)
model.compile(optimizer='SGD',loss='MSE',metrics=['accuracy'])
model.fit(x,one_hot_y,epochs=5)
#x_t=x_test.reshape((10000,784))
pred=model.predict(x)
c=0
for i in range(0,len(y)):
  if(np.argmax(pred[i])==y[i]):
    c+=1
print c
print 'Total accuracy ',(float(c)*100.00)/len(y)