import tensorflow as tf

def print_tensor(t,name):
    print(name)
    print('eval : ' + str(t.eval()))
    print('name : ' + t.name)
    print('shape : ' + str(t.shape))
    print('dtype : ' + str(t.dtype))
    print()

# Operator : Operator is Tensor.
const1 = tf.constant(2.0)
const2 = tf.constant(3.0)
mat1 = tf.constant([[1.0,2.0],[3.0,4.0]])
mat2 = tf.constant([[5.0,6.0],[7.0,8.0]])
ope1 = tf.add(const1,const2) # add
print(ope1)
ope2 = tf.subtract(const1,const2) # subtract
print(ope2)
ope3 = tf.multiply(mat1,mat2) # multiply of element
print(ope3)
ope4 = tf.scalar_mul(const1,mat1) # multiply of scalar
print(ope4)
ope5 = tf.matmul(mat1,mat2) # multiply of tensor
print(ope5)
ope6 = tf.divide(const1,const2) # divide
print(ope6)

with tf.Session():
    print_tensor(ope1,'ope1')
    print_tensor(ope2,'ope2')
    print_tensor(ope3,'ope3')
    print_tensor(ope4,'ope4')
    print_tensor(ope5,'ope5')
    print_tensor(ope6,'ope6')
