import tensorflow as tf

def print_tensor(t,name):
    print(name)
    print('eval : ' + str(t.eval()))
    print('name : ' + t.name)
    print('shape : ' + str(t.shape))
    print('dtype : ' + str(t.dtype))
    print()

# Variable : Changable value in dataflow graph.
var1 = tf.Variable(1)
print(var1)
var2 = tf.Variable(3.14)
print(var2)
var3 = tf.Variable([[1.0,2.0],[3.0,4.0]])
print(var3)
increment = tf.assign_add(var1,1)
decrement = tf.assign_sub(var1,1)

with tf.Session() as sess:
    #sess.run(tf.variables_initializer([var1])) # Variable must be initialized.
    sess.run(tf.global_variables_initializer()) # Variable must be initialized.
    print_tensor(var1,'var1')
    print_tensor(var2,'var2')
    print_tensor(var3,'var3')
    print('increment')
    increment.eval()
    print(var1.eval())
    print('decrement')
    decrement.eval()
    print(var1.eval())
