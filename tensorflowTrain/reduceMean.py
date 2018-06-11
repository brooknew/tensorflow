import tensorflow as tf
import numpy as np 

x = np.array( [  [2.0,3.] , [10. , 20. ] ] )
print(x)
print(type(x) )
xm = [ tf.reduce_mean(x) , tf.reduce_mean(x,0) , tf.reduce_mean(x,1) ] 

with tf.Session() as sess :
    r = sess.run(xm)
    print( r[0] )
    print( r[1] )
    print( r[2] ) 
