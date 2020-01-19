from __future__ import print_function
import tensorflow as tf

#import keras
#from keras.models import Sequential
#from keras.layers import Dense,Activation

import numpy as  np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

batch_size = 32
epochs = 5000

def func(x, y):
    return (x**2 + y**2)-1

x = np.arange(-1.0, 1.0, 0.05)
y = np.arange(-1.0, 1.0, 0.05)
X_, Y_ = np.meshgrid(x, y)
Z_ = func(X_, Y_)

#fig = plt.figure()
#ax = Axes3D(fig)
#ax.set_xlabel("x")
#ax.set_ylabel("y")
#ax.set_zlabel("f(x, y)")
#ax.plot_wireframe(X_, Y_, Z_)
#plt.show()

x_size = x.shape[0]
print('x_size : ',x_size)
y_size = y.shape[0]
print('y_size : ',y_size)

X = np.zeros((x_size*y_size,2))
print(X.shape)
print(X[0,0]) # indexは0始まり
print(X[0,1])
i = 0
for x_elem in x:
  for y_elem in y:
    X[i,0] = x_elem
    X[i,1] = y_elem
    i = i + 1
print('X[0,0] : ',X[1,0])
print('X[0,1] : ',X[1,1])

z = np.zeros(x_size*y_size)
i = 0
for xe in X:
  z[i] = func(xe[0],xe[1])
  i = i + 1
print(z.shape)
print('z[0] : ',z[0])

# トレーニング用データ/検証用データの分割
X_train,X_test,y_train,y_test = train_test_split(X,z,train_size=0.8) # トレーニング用データ80%で分割

# model difinition
#model = Sequential()
##model.add(Dense(3,input_shape=(1,)))
##model.add(Activation('sigmoid'))
##model.add(Activation('relu'))
#model.add(Dense(6,input_shape=(1,)))
#model.add(Activation('sigmoid'))
#model.add(Dense(1))
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2,)),
  tf.keras.layers.Dense(3,activation='sigmoid'),
  tf.keras.layers.Dense(6),
  tf.keras.layers.Dense(1)
])
model.summary()

model.compile(loss='mean_squared_error',#tf.keras.losses.mean_squared_error(),
             optimizer='Adam',#tf.keras.optimizers.RMSprop(),
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

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")
#ax.plot_wireframe(X_test[:,0], X_test[:,1], y_predict)
ax.scatter(X_test[:,0], X_test[:,1], y_predict)
plt.show()
