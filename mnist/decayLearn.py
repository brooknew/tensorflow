import tensorflow as tf;  
import numpy as np;  
import matplotlib.pyplot as plt;  
  
learning_rate = 0.1  
decay_rate = 0.96  
global_steps = 1000  
decay_steps = 100  
  
global_ = tf.Variable(tf.constant(0))
global_step = tf.Variable(1, trainable=False)

#c = tf.train.exponential_decay(learning_rate, global_ = global_step , decay_steps = 100 , decay_rate = 0.96, staircase=True/False)

c = tf.train.exponential_decay(learning_rate, global_ ,decay_steps, decay_rate, staircase=True)  
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)  
  
T_C = []  
F_D = []  
  
with tf.Session() as sess:
    F_d = sess.run(d,feed_dict={global_: 1})
    print( 'F_d=' , F_d ) 
    for i in range(global_steps):  
        T_c = sess.run(c ,feed_dict={global_: i})  
        T_C.append(T_c)  
        F_d = sess.run(d,feed_dict={global_: i})  
        F_D.append(F_d)  
  
  
plt.figure(1)  
plt.plot(range(global_steps), F_D, 'r-')  
plt.plot(range(global_steps), T_C, 'b-')  
      
plt.show()  
