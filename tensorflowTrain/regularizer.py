#tensorboard --logdir=log_simple_status

import tensorflow as tf

def main() :
    a =tf.constant(  [ 1., 2., 3. , 100] ) 
    regf = tf.contrib.layers.l2_regularizer(3.)
    print( type ( regf ) ) 
    reg = regf( a )
    print( reg )
    print ( type ( reg ) )
    sess = tf.Session()
    r = sess.run( reg )
    print( type( r ) )
    print( r )
    

main()
