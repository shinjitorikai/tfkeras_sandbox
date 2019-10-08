import tensorflow as tf

const1 = tf.constant(1.2)
const2 = tf.constant(3.4)
ans1 = tf.add(const1,const2)
ans2 = tf.subtract(const1,const2)

with tf.Session() as sess: # create session instance
    # evaluation one node in flow graph
    print(ans1.eval())

    # evaluation multi node in flow graph
    ans = sess.run([ans1,ans2])
    print(ans)
