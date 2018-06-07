import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
from saveTensor import *

TRAIN_STEPS = 100000
LAYER1_NODE = 500 
REGULARIZER = 0.0001
LEARN_RATE = 0.1
LEARNRATE_START =  0.1
DECAPY_STEPS  = 200 #50000/100
DECAY_RATE = 0.99

LAYER=2


def trainMain( ):
    print("start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    w1 = tf.Variable( tf.zeros( [784, LAYER1_NODE] ) )
    b1 = tf.Variable( tf.zeros([LAYER1_NODE]) )
    y1 =  tf.nn.relu( tf.matmul(x,w1) + b1)
    w2 = tf.Variable( tf.zeros( [LAYER1_NODE,10] ) )
    b2 = tf.Variable( tf.zeros([10]) )
    w3 = tf.Variable( tf.zeros( [784,10] ) )
    y2 = tf.matmul(x,w3)
    #tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(REGULARIZER)(w3))
    if LAYER == 2 :
        y = tf.nn.softmax(  tf.matmul(y1,w2) + b2 )
    else :
        y = tf.nn.softmax( y2 + b2 )
    y_=tf.placeholder('float',[None,10] )
    cross_entropy = -tf.reduce_sum( y_*tf.log(y) ) # + tf.add_n(tf.get_collection('losses') )
    global_stepL = tf.Variable( 0 , trainable = False  )
    learnRate = tf.train.exponential_decay( LEARNRATE_START , global_stepL , DECAPY_STEPS , DECAY_RATE, staircase = True) 
    train_step = tf.train.GradientDescentOptimizer(learnRate).minimize( cross_entropy , global_step = global_stepL )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run( init )
    saver = tf.train.Saver() 
    for i in range(TRAIN_STEPS):
        batch_xs , batch_ys = mnist.train.next_batch( 100 )
        sess.run( train_step , feed_dict={x:batch_xs , y_:batch_ys} )
        if  i%1000 == 0 :
            sampNum = 20000 
            batch_xs , batch_ys = mnist.train.next_batch( sampNum )
            cross_entropyR = sess.run( cross_entropy  ,  feed_dict={x:batch_xs , y_:batch_ys} )
            cross_entropyR /= ( sampNum/100 )
            lr = sess.run( learnRate )
            gsv = sess.run( global_stepL ) 
            print( i  , " loss=" , cross_entropyR , 'global_step:' , gsv , 'learn rate:' , lr  )

    saver.save( sess , 'model/checkpt' )    
    if LAYER == 1 :
        rb2 = sess.run( b2 )
        rw3 = sess.run( w3 )        
        saveMatrix( rw3 ,'w3',   'w_b.txt' , 'wt' )
        saveMatrix( rb2 , 'b2' , 'w_b.txt' , 'at' )
    elif LAYER == 2 :
        rw1 = sess.run( w1 )
        rb1 = sess.run( b1 )
        rw2 = sess.run( w2 )
        rb2 = sess.run( b2 )
        saveMatrix( rw1 ,'w1',   'w_b_layer2.txt' , 'wt' )
        saveMatrix( rb1 , 'b1' , 'w_b_layer2.txt' , 'at' )
        saveMatrix( rw2 ,'w2',   'w_b_layer2.txt' , 'at' )
        saveMatrix( rb2 , 'b2' , 'w_b_layer2.txt' , 'at' )
    print( "train end" )
    
def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    trainMain()

if __name__ == '__main__':
    main( )


def testMain() :
    print( "test start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    w1 = tf.Variable( tf.zeros( [784,LAYER1_NODE] ) )
    b1 = tf.Variable( tf.zeros([LAYER1_NODE]) )
    y1 = tf.nn.relu( tf.matmul(x,w1) + b1)
    w2 = tf.Variable( tf.zeros( [LAYER1_NODE,10] ) )
    b2 = tf.Variable( tf.zeros([10]) )
    w3 = tf.Variable( tf.zeros( [784,10] ) )
    if LAYER == 2 :
        y = tf.nn.softmax( tf.matmul(y1,w2) + b2 )
    else :
        y = tf.nn.softmax( tf.matmul(x,w3) + b2 )
    y_=tf.placeholder('float',[None,10] )
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    sess = tf.Session()
    sess.run( init )
    ckpt = tf.train.get_checkpoint_state( 'model/')
    if ckpt and ckpt.model_checkpoint_path:
        print(  ckpt.model_checkpoint_path ) 
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
    correct_prediction = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
    accuracy = tf.reduce_mean( tf.cast( correct_prediction , 'float')  )
    print( sess.run( accuracy , feed_dict={x:mnist.train.images[500:600] , y_:mnist.train.labels[500:600]} ) )

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

                        
