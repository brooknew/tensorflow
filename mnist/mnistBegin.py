import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 

def trainMain( Step):
    print("start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    w = tf.Variable( tf.zeros( [784,10] ) )
    b = tf.Variable( tf.zeros([10]) )
    y = tf.nn.softmax( tf.matmul(x,w) + b )
    y_=tf.placeholder('float',[None,10] )
    cross_entropy = -tf.reduce_sum( y_*tf.log(y) )
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize( cross_entropy )
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run( init )
    saver = tf.train.Saver() 
    for i in range(Step):
        batch_xs , batch_ys = mnist.train.next_batch( 100 )
        sess.run( train_step , feed_dict={x:batch_xs , y_:batch_ys} )
    saver.save( sess , 'model/checkpt' ) 
    print( "train end" )

def testMain() :
    print( "test start" )
    mnist = input_data.read_data_sets('data/',one_hot=True )
    x = tf.placeholder( 'float' , [None , 784] )
    w = tf.Variable( tf.zeros( [784,10] ) )
    b = tf.Variable( tf.zeros([10]) )
    y = tf.nn.softmax( tf.matmul(x,w) + b )
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
    print( sess.run( accuracy , feed_dict={x:mnist.test.images , y_:mnist.test.labels} ) )
    print( "test end" ) 


                        
