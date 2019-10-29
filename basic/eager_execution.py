from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

# EagerExecutionの状態(True:有効、False:無効)
tf.compat.v1.enable_eager_execution() # 有効に切り替え
print('TensorFlow Eagerly is : ' + str(tf.executing_eagerly()))

# TensorFlowのオペレーションを実行
x = [[1.23]]
m = tf.matmul(x,x)
print(m)
print('{}'.format(m))
