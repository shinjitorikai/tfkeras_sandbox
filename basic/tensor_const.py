import tensorflow as tf

def print_tensor(t,name):
    print(name)
    print('eval : ' + str(t.eval()))
    print('name : ' + t.name)
    print('shape : ' + str(t.shape))
    print('dtype : ' + str(t.dtype))
    print()

# Constant : immutable value in Dataflow graph.
const1 = tf.constant(1.23) # constant
print(const1)
const2 = tf.zeros([2,3]) # zero constant
print(const2)
const3 = tf.ones([2,1]) # one constant
print(const3)

with tf.Session():
    print_tensor(const1,'const1')
    print_tensor(const2,'const2')
    print_tensor(const3,'const3')
