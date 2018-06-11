import numpy as np
import tensorflow as tf

SIZE=6
CLASS=10
label1=np.random.randint(0,10,size=SIZE) 
print( label1)
b = tf.one_hot(label1,CLASS,1,0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(b)
    print(sess.run(b))
