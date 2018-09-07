import numpy as np 
import tensorflow as tf

anp = np.array( [ [[1,2,3,4],[5,6,7,8]], [[10,20,30,40],[50,60,70,80]]] )
a = tf.Variable( anp ) 
ashape = a.get_shape().as_list()

res = tf.reshape( a , [ashape[0] , ashape[1]*ashape[2]] ) 
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run( init )
ar = sess.run( a )
resr = sess.run(res)
#print(anp)
#print(ar)
print( 'resr' , resr ) 
