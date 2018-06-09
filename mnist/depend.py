import tensorflow as tf

a = tf.Variable( 2 )
selfAdd = tf.Variable( 0 ) 
selfAddition = tf.assign_add(  selfAdd , 3    )
selfSub = tf.Variable( 0 ) 
selfSubtraction = tf.assign_sub(  selfSub , 2    )

b = tf.multiply( a , selfAdd )

with  tf.control_dependencies( [ selfAddition ] ) :
    train_op  =  tf.no_op()
    print( train_op ) 
    train_op  =[ train_op ,  selfSubtraction]
    print( train_op )
    
with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run( init )
    for i in range( 20 ) :
        sess.run( train_op )
        print( "selfAdd:" , sess.run(  selfAdd ) )

    ra = sess.run(selfAdd )
    rb = sess.run(b )
    print( '@end selfAdd:' ,ra )
    rs = sess.run(selfSub)
    print('@end selfSub:' , rs ) 
    print( 'b:' , rb )
 

