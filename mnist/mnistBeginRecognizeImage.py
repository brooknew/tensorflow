import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def pre_pic(picName):
    img = Image.open( picName)
    reIm = img.resize( (28,28) , Image.ANTIALIAS )
    imgGray = reIm.convert('L')
    arr = np.array(  imgGray )    
    nm_arr_0 = arr.reshape([784] )
    nm_arr = np.empty( (1,784) ,  dtype =  np.float32  ) 
    for i  in range( 784 ) :
        pix = (255.0 - nm_arr_0[i])/255.0
        if pix < 40.0/256:
            pix = 0.0
        else:
            pix = 1.0
        nm_arr[0][i] = pix
    return nm_arr

def  recognizeImageMain():
    print( "recognizeImage start" )
    x = tf.placeholder( 'float' , [1, 784] )
    w = tf.Variable( tf.zeros( [784,10] ) )
    b = tf.Variable( tf.zeros([10]) )
    y = tf.nn.softmax( tf.matmul(x,w) + b )
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
        mnistimg =  np.array(  testPicArr[0] ) 
        img =  mnistimg.reshape ( 28 ,28 )
        plt.imshow(  img ,  cmap = 'gray', interpolation = 'bicubic' )
        plt.show()


recognizeImageMain()
