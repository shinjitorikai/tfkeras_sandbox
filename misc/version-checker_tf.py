import tensorflow as tf
import tensorflow_hub as hub

print('TensorFlow ver', tf.__version__)
print('TensorFlow Hub ver', hub.__version__)
print('Eager mode : ', tf.executing_eagerly())
if tf.config.experimental.list_physical_devices('GPU'):
    print('GPU : available')
else:
    print('GPU : not available')
