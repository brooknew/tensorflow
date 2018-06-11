import tensorflow as tf

def argmax1( ):
    a = [1,5,0]
    argC =[  tf.argmax( a , 0 ) , tf.argmax( a , 0 ) ]
      
    with tf.Session() as se :
        r = se.run(argC )
        print( r[0] )
        print ( r[1] )

def findMaxFromRank3() :
    """
    找出一个三阶张量第三维(索引是2）的最大值
    """
    a =tf.Variable( [[[10.0,25.0,3.0,4.0] , [10.0,251.0,35.0,4.0]] , [[100.0,25.0,3.0,4.0] , [10.0,250.0,3500.0,4.0]] ] )
    b = tf.argmax(  a  , 2 )
    se = tf.Session()
    init = tf.global_variables_initializer()
    se.run( init ) 
    r = se.run( b )
    ar = se.run( a )
    se.close()
    print( r )
    print ( type (r )  )
    print( ar[0][0][r[0][0]] , "," , ar[0][1][ r[0][1] ] )
    print( ar[1][0][r[1][0]] , "," , ar[1][1][ r[1][1] ] )

def findMaxFromRank1() :
    """
    找出一个一阶张量第一维(索引是0）的最大值
    """
    a =tf.Variable( [10.0,250.0,3510.0,4.0 ] )
    b = tf.argmax(  a  , 0 )
    se = tf.Session()
    init = tf.global_variables_initializer()
    se.run( init ) 
    r = se.run( b )
    ar = se.run( a )
    se.close()
    print( r )
    print ( type (r )  )
    print( ar[r] )
    

findMaxFromRank3()
findMaxFromRank1() 
argmax1( )
