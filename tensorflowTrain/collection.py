#tensorboard --logdir=log_simple_status

import tensorflow as tf
import numpy as np 

def setCollection (t , Lev):
    if  (Lev == 'l1'): 
        regf = tf.contrib.layers.l1_regularizer( 0.1 )
    else:
        regf = tf.contrib.layers.l2_regularizer( 0.1 )
          
    reg = regf(  t ) 
    tf.add_to_collection( Lev  ,reg )


def getCollection( Lev ):
    reg = tf.get_collection( Lev )
    return reg

def main() :
    sess = tf.Session()
    aa  =  [ [1.,2.,3.,4.,5.] , [1.,2.,30.,4.,5.]]  
    a = tf.placeholder( tf.float32 ,  [5] )
    #sess.run( tf.global_variables_initializer () )
    setCollection (a , 'l1')
    setCollection (a , 'l2')
    layers = [ 'l1' , 'l2' ]
    for L in layers :
        reg = getCollection( L )
        print( reg )
        i = 0 
        while i < 2 :
            r = sess.run( reg ,  feed_dict={a:aa[i] }  )
            print( type( r ) )
            print( r )
            i = i + 1
 
main()
