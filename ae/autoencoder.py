# AutoEncoder by TensorFlow2
#  - dataset : MNIST

import tensorflow as tf
import matplotlib.pyplot as plt

EPOCH = 5

# load mnist dataset
print('load mnist dataset...')
mnist = tf.keras.datasets.mnist
(x_train, _),(x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# define model
model = tf.keras.models.Sequential([
    # Encoder
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    # Decoder
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28,28))
])
print('model summary:')
print(model.summary())

# complie and fit
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
history = model.fit(x_train, x_train, epochs=EPOCH)
plt.plot(history.history['loss'],label='loss')
plt.plot(history.history['accuracy'],label='acc')
plt.legend()
plt.show()
result = model.evaluate(x_test, x_test)
print('loss :',result[0])
print('acc  :',result[1])

# show prediction example
pred_num = 10
pred = model.predict(x_test[0:pred_num])

plt_row = 2
plt_col = pred_num
plt.figure(figsize=(28,5))
for i in range(pred_num):
  plt.subplot(plt_row,plt_col,i+1)
  plt.imshow(x_test[i])
  plt.axis('off')
for i in range(pred_num):
  plt.subplot(plt_row,plt_col,i+1+pred_num)
  plt.imshow(pred[i])
  plt.axis('off')
plt.show()
