import tensorflow as tf



def  func1() :
    var1 = tf.Variable( 56 , name = 'var1' )
    var2 = tf.Variable( 6 , name = 'var2' )
    print( 'var1 inited :' , var1 )
    selfAdd = tf.assign_add( var1 , 2 )
    return selfAdd 


def func2(  sessI ):
    for  var in  tf.global_variables() :
        #print ( var )
        if ( var.name == 'var1:0' ) :
            print (var.name , sessI.run( var ) ) 


op1 = func1()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run( init )
    for i in range(10) :
        sess.run( op1 ) 
        func2 (sess )
    
