import tensorflow as tf
from tensorflow.python.client import device_lib

print('TensorFlow ver', tf.__version__)
print('Eager mode : ', tf.executing_eagerly())
if tf.config.experimental.list_physical_devices('GPU'):
    print('GPU : available')
else:
    print('GPU : not available')

print('local devices:')
print(device_lib.list_local_devices())
