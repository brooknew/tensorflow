import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mnistBegin import *

THRESH = 50

def pre_pic(picName):
    img = Image.open( picName)
    reIm = img.resize( (28,28) , Image.ANTIALIAS )
    imgGray = reIm.convert('L')
    arr = np.array(  imgGray )    
    nm_arr_0 = arr.reshape([784] )
    nm_arr = np.empty( (1,784) ,  dtype =  np.float32  ) 
    for i  in range( 784 ) :
        pix = (255  - nm_arr_0[i])
        if pix < THRESH:
            pixv = 0.0
        else:
            pixv = 1.0
        nm_arr[0][i] = pixv
    return nm_arr

def  recognizeImageMain():
    print( "recognizeImage start" )
    x = tf.placeholder( 'float' , [None , 784] )
    w1 = tf.Variable( tf.zeros( [784, LAYER1_NODE] ) )
    b1 = tf.Variable( tf.zeros([LAYER1_NODE]) )
    y1 = tf.nn.relu( tf.matmul(x,w1) + b1)
    w2 = tf.Variable( tf.zeros( [LAYER1_NODE,10] ) )
    b2 = tf.Variable( tf.zeros([10]) )
    w3 = tf.Variable( tf.zeros( [784,10] ) )
    if LAYER == 2 :
        y = tf.nn.softmax( tf.matmul(y1,w2) + b2 )
    else:
        y = tf.nn.softmax( tf.matmul(x,w3) + b2 )
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    sess = tf.Session()
    sess.run( init )
    ckpt = tf.train.get_checkpoint_state( 'model/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    testNum = int (input("input the number of test pictures:"))
    for i in range(testNum):
        testPic = input("the path of test picture:")
        testPicArr = pre_pic(testPic)
        yr = sess.run( y  , feed_dict={x:testPicArr } )
        finddigit = tf.argmax( yr , 1  )
        ind = sess.run( finddigit )
        print( "digit is : " , ind[0] )
        print( "yr:" , yr[0][ind[0]] )
        print ( yr ) 
        mnistimg =  np.array(  testPicArr[0] ) 
        img =  mnistimg.reshape ( 28 ,28 )
        plt.imshow(  img ,  cmap = 'gray', interpolation = 'bicubic' )
        plt.show()


recognizeImageMain()
