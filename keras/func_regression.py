from __future__ import print_function
import tensorflow as tf

#import keras
#from keras.models import Sequential
#from keras.layers import Dense,Activation

import numpy as  np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

batch_size = 32
epochs = 5000

#def func(x): # 1次
#  a = 0.2 # ポイント：データが0<y<1に収まるように比率を調整
#  b = 0.3
#
#  y = a*x + b
#  
#  return y

#def func(x): # 2次
#  a = 0.45 # ポイント：データが0<y<1に収まるように比率を調整
#  b = -0.35
#  c = 0.15
#  y = a*x*x + b*x + c
#  
#  return y

def func(x):
  a = 0.5
  b = -0.15
  c = -0.2
  d = 0.7
  y = a*x*x*x + b*x*x + c*x + d
  
  return y


X = np.linspace(-1,1,1000)
y = func(X)

# show func(x)
#plt.scatter(X,y)
#plt.grid(True)
#plt.xlabel('X')
#plt.ylabel('y')
#plt.show()

# トレーニング用データ/検証用データの分割
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8) # トレーニング用データ80%で分割

# model difinition
#model = Sequential()
##model.add(Dense(3,input_shape=(1,)))
##model.add(Activation('sigmoid'))
##model.add(Activation('relu'))
#model.add(Dense(6,input_shape=(1,)))
#model.add(Activation('sigmoid'))
#model.add(Dense(1))
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1,)),
  tf.keras.layers.Dense(3,activation='sigmoid'),
  tf.keras.layers.Dense(1)
])
model.summary()

model.compile(loss='mean_squared_error',#tf.keras.losses.mean_squared_error(),
             optimizer='RMSProp',#tf.keras.optimizers.RMSprop(),
             metrics=['accuracy'])

history = model.fit(X_train,y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(X_test,y_test))

score = model.evaluate(X_test,y_test,verbose=0)
test_loss = score[0]
test_accuracy = score[1]
print('Test loss : ', test_loss)
print('Test accuracy : ', test_accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid(True)
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'])
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid(True)
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])
plt.show()

y_predict = model.predict(X_test)

#plt.plot(X_test,y_predict)
plt.scatter(X_test,y_predict)
#plt.plot(X_test,y_test)
plt.scatter(X_test,y_test)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['predict', 'test'])
plt.show()
