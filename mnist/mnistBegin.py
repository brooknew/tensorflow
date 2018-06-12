import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
from saveTensor import *

#([784,500]) + b1[500] -> ([500,10])+b2[10]  ,  Accuracy Rate is as the following
# 1 no     regularizer : 0.9603
# 2 have  regularizer : 0.9593

TRAIN_STEPS = 100000 
LAYER1_NODE = 500 
REGULARIZER = 0.0001
REGUL_COLLECTION='loss'
LEARN_RATE = 0.1
LEARNRATE_START =  0.1
DECAPY_STEPS  = 200 #50000/100
DECAY_RATE = 0.99

LAYER=2

def forward( x ):
    if LAYER == 2 :
        w1 = tf.Variable( tf.truncated_normal( [784, LAYER1_NODE] ,stddev=0.1)  , name='w1' ) #tf.zeros( [784, LAYER1_NODE] ) )
        tf.add_to_collection( REGUL_COLLECTION ,  tf.contrib.layers.l2_regularizer(REGULARIZER)(w1) ) 
        b1 = tf.Variable( tf.zeros([LAYER1_NODE]) , name='b1' )
        y1 =  tf.nn.relu( tf.matmul(x,w1) + b1)
        w2 = tf.Variable( tf.truncated_normal(  [LAYER1_NODE,10]  ,stddev=0.1)  , name = 'w2' )   # tf.zeros( [LAYER1_NODE,10] ) )
        tf.add_to_collection( REGUL_COLLECTION ,  tf.contrib.layers.l2_regularizer(REGULARIZER)(w2) )
        b2 = tf.Variable( tf.zeros([10]) , name = 'b2' )         
        y = tf.nn.softmax(  tf.matmul(y1,w2) + b2 )
    else :
        b2 = tf.Variable( tf.zeros([10]) , name = 'b2' )
        w3 = tf.Variable( tf.zeros( [784,10] )  , name = 'w3')
        tf.add_to_collection( REGUL_COLLECTION ,  tf.contrib.layers.l2_regularizer(REGULARIZER)(w3) )
        y2 = tf.matmul(x,w3)        
        y = tf.nn.softmax( y2 + b2 )
    return y 

def saveVar( sess ): 
    if LAYER == 1 :
        for var in tf.global_variables() :
            if( var.name =='b2:0' ):                 
                rb2 = sess.run( var )
                saveMatrix( rb2 , 'b2' , 'w_b.txt' , 'at' )
            elif ( var.name == 'w3:0') :
                rw3 = sess.run( w3 )        
                saveMatrix( rw3 ,'w3',   'w_b.txt' , 'wt' )        
    elif LAYER == 2 :
        for var in tf.global_variables() :
            if( var.name =='w1:0' ):  
                rw1 = sess.run( var )
                saveMatrix( rw1 ,'w1',   'w_b_layer2.txt' , 'wt' )
            elif var.name =='b1:0' :
                rb1 = sess.run( var )
                saveMatrix( rb1 , 'b1' , 'w_b_layer2.txt' , 'at' )
            elif var.name == 'w2:0' :
                rw2 = sess.run( var )
                saveMatrix( rw2 ,'w2',   'w_b_layer2.txt' , 'at' )
            elif var.name == 'b2:0':
                rb2 = sess.run( var )
                saveMatrix( rb2 , 'b2' , 'w_b_layer2.txt' , 'at' )
        rw1abs = abs( rw1 )
        rb1abs = abs( rb1 )
        rw2abs= abs( rw2 )
        rb2abs = abs( rb2 )
        w1sum = tf.reduce_sum( rw1abs )
        b1sum = tf.reduce_sum( rb1abs )
        w2sum = tf.reduce_sum( rw2abs )
        b2sum = tf.reduce_sum( rb2abs )
        print( 'w1sum:' , sess.run(w1sum) )
        print( 'b1sum:' , sess.run(b1sum ))
        print( 'w2sum:' , sess.run(w2sum ))
        print( 'b2sum:' , sess.run(b2sum ) )



def trainMain( ):
    print("start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    y = forward( x )
    y_=tf.placeholder('float',[None,10] )
    cross_entropy = -tf.reduce_mean( y_*tf.log(y) ) #
    if( LAYER == 2 ) :
        cross_entropy +=   tf.add_n(tf.get_collection(REGUL_COLLECTION) )
    #cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #cross_entropy = tf.reduce_mean( cross_ent )  
    global_stepL = tf.Variable( 0 , trainable = False  )
    learnRate = tf.train.exponential_decay( LEARNRATE_START , global_stepL , DECAPY_STEPS , DECAY_RATE, staircase = True) 
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize( cross_entropy , global_step = global_stepL )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run( init )
    for var in tf.global_variables():
        print(var.name)
    saver = tf.train.Saver() 
    for i in range(TRAIN_STEPS):
        batch_xs , batch_ys = mnist.train.next_batch( 100 )
        _,  cross_entropyR =  sess.run( [ train_step , cross_entropy],  feed_dict={x:batch_xs , y_:batch_ys} )
        if  i%10000 == 0 :
            lr = sess.run( learnRate )
            gsv = sess.run( global_stepL ) 
            print( i  , " loss=" , cross_entropyR , 'global_step:' , gsv , 'learn rate:' , lr  )

    saver.save( sess , 'model/checkpt' )
    
    correct_prediction = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction , 'float')  )
    print( "Accuracy Rate:" , sess.run( accuracy , feed_dict={x:mnist.test.images[0:10000] , y_:mnist.test.labels[0:10000]} ) )
    
    saveVar( sess ) 
    print( "train end" )
    

def testMain() :
    print( "test start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    y = forward( x )
    y_=tf.placeholder('float',[None,10] )
    saver = tf.train.Saver() 
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run( init )

    ckpt = tf.train.get_checkpoint_state( 'model/')
    if ckpt and ckpt.model_checkpoint_path:
        print(  ckpt.model_checkpoint_path ) 
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    correct_prediction = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction , 'float')  )
    print( "Accuracy Rate:" ,sess.run( accuracy , feed_dict={x:mnist.test.images[0:10000] , y_:mnist.test.labels[0:10000]} ) )

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    trainMain()
    #testMain()

if __name__ == '__main__':
    main( )


def recognizeOne() :
    print( "recognize start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    w1 = tf.Variable( tf.zeros( [784,LAYER1_NODE] ) )
    b1 = tf.Variable( tf.zeros([LAYER1_NODE]) )
    y1 = tf.nn.relu( tf.matmul(x,w1) + b1)
    w2 = tf.Variable( tf.zeros( [LAYER1_NODE,10] ) )
    b2 = tf.Variable( tf.zeros([10]) )
    w3 = tf.Variable( tf.zeros( [784,10] ) )
    y = tf.nn.softmax( tf.matmul(y1,w2) + b2 )
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    sess = tf.Session()
    sess.run( init )
    ckpt = tf.train.get_checkpoint_state( 'model/')
    if ckpt and ckpt.model_checkpoint_path:
        print(  ckpt.model_checkpoint_path ) 
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    
    n = int(input("please enter the digit image No to test\n" ))
    yr = sess.run( y  , feed_dict={x:mnist.test.images[n:n+1]}  )
    finddigit = tf.argmax( yr , 1  )
    ind = sess.run( finddigit )
    
    print( "digit is : " , ind[0] )#, "prob is : " ,  yr[ ind ] )
    print( "yr is : " ,  yr [0] )  
    y_ = mnist.test.labels[n]
    ind1  = tf.argmax( y_)
    ind1r = sess.run( ind1 )
    print(  y_[ind1r] ) 
    diff = y_[ind1r] - yr[0][ind[0] ]
    diff = abs ( diff )
    if  diff  > 0.1 :
        print ( "fail" )
    else :
        print( "right" )

    mnistimg =  np.array(   mnist.test.images[n] ) 
    img =  mnistimg.reshape ( 28 ,28 )
    plt.imshow(  img ,  cmap = 'gray', interpolation = 'bicubic' )
    plt.show()
    
    print( "test end" ) 

                        
