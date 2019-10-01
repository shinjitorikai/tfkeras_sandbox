import tensorflow as tf

def print_tensor(t,name):
    print(name)
    print('eval : ' + str(t.eval(feed_dict={t:1.23})))
    print('name : ' + t.name)
    print('shape : ' + str(t.shape))
    print('dtype : ' + str(t.dtype))
    print()

# PlaceHolder : PlaceHolder is Tensor.
ph1 = tf.placeholder(tf.float32)
print(ph1)
ph2 = tf.placeholder_with_default(tf.constant(2.0),[]) # with default value
print(ph2)

with tf.Session():
    print_tensor(ph1,'ph1')
    print_tensor(ph2,'ph2')
