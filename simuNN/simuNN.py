#coding:utf-8
import tensorflow as tf
import numpy as np
from traindata import *
from dataDim import *

BATCH_SIZE = 8
TRAIN_STEPS = 100000

def  matrixCopy( m , mNew ) :
    for  r  in   range( len( m ) )  :
        for c in  range ( len ( m[r] ) ) :
            mNew[r][c] = m[r][c]
def forwardnn(xa, w1 , w2 ,b1,b2, ya) :
    i = 0 
    for  x in xa :
        a1 = np.dot( x , w1 )
        a1 += b1
        y = np.dot( a1 , w2 )
        y += b2
        ya[ i ] = y
        i = i + 1
def  squareErrorAvg( y_a , ya ) :
    e = 0.0
    yindex = 0 ;
    for  y_    in   y_a  :
        y = ya[yindex]
        e1 = (y_[0]- y[0]) * (y_[0]- y[0])
        e  += e1
        yindex = yindex + 1
    e  /=  yindex
    return e 

def  learn( layer ,  w1, w2 , b1 , b2 , wnew ,  xa , ya , y_a , learnRate , wDelta  ) :
    forwardnn( xa ,  w1 , w2 ,b1,b2, ya )
    e1 = squareErrorAvg( y_a , ya )
    if layer == 1 :
        w = w1
    else:
        w = w2 
    for  r  in   range( len( wnew ) )  :
        for c in  range ( len ( wnew[r] ) ) :
            v = w[r][c]
            w[r][c] += wDelta
            forwardnn( xa ,  w1 , w2 , b1 ,b2 , ya )
            e2 = squareErrorAvg( y_a , ya )
            learnN = (e2-e1)/wDelta * learnRate # (e2-e1)/wDelta :loss函數的微分 
            w[r][c] =  v
            wnew[r][c]  =  v- learnN
            
def  learnB( layer ,  w1, w2 , b1 ,b2 , bnew ,  xa , ya , y_a , learnRate , wDelta  ) :
    forwardnn( xa ,  w1 , w2 ,b1,b2, ya )
    e1 = squareErrorAvg( y_a , ya )
    if layer == 1 :
        b = b1
    else:
        b = b2     
    for c in  range ( len ( bnew) ) :
        v = b[c]
        b[c] += wDelta
        forwardnn( xa ,  w1 , w2 , b1, b2 , ya )
        e2 = squareErrorAvg( y_a , ya )
        learnN = (e2-e1)/wDelta * learnRate # (e2-e1)/wDelta :loss函數的微分 
        b[c] =  v
        bnew[c]  =  v- learnN

def calAccury(  ya ) : # ya is list to predicted result   
    ya = np.array( ya )
    y_a = Y_T
    y_a = np.array( y_a ) 
    accury =np.array(ya - y_a)
    a = ( abs( accury ) < TRAIN_THRESHOLD )
    af = a.astype( np.float32 )
    right = af.sum()
    per = right/VALIDATE_SIZE 
    return per , a 

def directValidateMain(w1 , w2 , b1 ,b2 ):
    xa = XT[0:VALIDATE_SIZE]
    ya =  [[0.0]]* VALIDATE_SIZE
    forwardnn(xa, w1 , w2 , b1 ,b2 , ya)
    per  , a  = calAccury(  ya )
    for i in range( len( ya ) ):
        print ( xa[i] , '=>'  , ya[i] , '||', Y_T[i] , a[i]  )          
    return per 

''' Main for  Leaned by myself '''
def directLearnMain() :
    w1= np.array( [[1.0,1.0,1.0],[2.0,1.0,1.0]])
    w2= np.array([[2.0],[3.0],[5.0]])
    #w1= np.array(  [[0.1,0.5,0.0],[0.0,0.0,0.0]] ) 
    #w2= np.array( [[0.05],[0.01],[0.05]] ) 
    b1=[0.0,0.0,0.0]
    b2=[0.0]

    Y = [    [0.0] ]* SAMPLE_SIZE 
         
    STEPS =TRAIN_STEPS
    w1new = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    w2new= [[0.0],[0.0],[0.0]]
    b1new=[0.0,0.0,0.0]
    b2new=[0.0]

    for i in range(STEPS):
        start =  (i*BATCH_SIZE) % SAMPLE_SIZE 
        end =  start + BATCH_SIZE
        xa = X[start:end]
        ya = Y[start:end]
        y_a = Y_[start:end]
        learn( 1 ,  w1, w2 , b1,b2, w1new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learn( 2 ,  w1, w2 , b1,b2 , w2new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learnB( 1 ,  w1, w2 , b1,b2, b1new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learnB( 2 ,  w1, w2 , b1,b2, b2new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        matrixCopy( w1new ,    w1)
        matrixCopy( w2new ,    w2)
        b1= b1new[:]
        b2=b2new[:]
        if  (i +1)% 10000  == 0 :
            forwardnn( xa ,  w1 , b1,b2 , w2 , ya )
            e = squareErrorAvg( y_a  , ya )
            print ('w1 learned  by myself @time(s)  ' , i+1 , '  : ' ,  w1new )
            print ('w2 learned  by myself @time(s)  ' ,  i +1, '  : '  ,  w2new )
            print('loss  by myself @time(s)  ' ,  i +1, '  : '   , e) 
    print( "训练结果(myself)：" )
    print("w1:", w1 )
    print("w2:", w2 )
    print( "b1" , b1 )
    print("b2" ,b2)
    per  = directValidateMain( w1 , w2  , b1 ,b2)
    print("Accury by myslef:" , per ) 
    return w1,w2, b1 , b2

''' main for training using tensorflow '''
def  tensorflowMain() :    
    #1定义神经网络的输入、参数和输出,定义前向传播过程。
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_= tf.placeholder(tf.float32, shape=(None, 1))

    w1= tf.Variable( [[1.0,1.0,1.0],[2.0,1.0,1.0]])
    w2= tf.Variable([[2.0],[3.0],[5.0]])
    #w1= tf.Variable( [[0.1,0.5,0.0],[0.0,0.0,0.0]])
    #w2= tf.Variable([[0.05],[0.01],[0.05]])
    b1 = tf.Variable( [0.0 , 0.0 , 0.0 ] )
    b2 = tf.Variable( [0.0  ] ) 
    a = tf.matmul(x, w1) + b1
    y = tf.matmul(a, w2) + b2 

    #2定义损失函数及反向传播方法。
    loss_mse = tf.reduce_mean(tf.square(y-y_)) 
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
    #3生成会话，训练STEPS轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
          
        # 训练模型。
        STEPS = TRAIN_STEPS
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % SAMPLE_SIZE 
            end =  start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
            if  (i+1) % 10000 == 0  :
                total_loss = sess.run(loss_mse, feed_dict={x: X[start:end], y_: Y_[start:end]})                
                print ("w1 after trained " , i+1 , " time(s)  by tensorflow:\n", sess.run(w1))
                print ("w2 after trained " , i +1, " time(s)  by tensorflow:\n", sess.run(w2))
                print("After %d training step(s), loss_mse  is %g" % (i+1, total_loss))
        print( "训练结果(Tensorflow)：" )
        r_w1 = sess.run(w1)
        r_w2 = sess.run(w2)
        r_b1 = sess.run( b1)
        r_b2 = sess.run( b2 )
        print("w1:", r_w1  )
        print("w2:", r_w2 )
        print ("b1:", r_b1 )
        print ("b2:", r_b2 )

        #validate
        rv = sess.run( y , feed_dict={x:XT } )
        per , accAr = calAccury( rv )
        print("Tensorflow accury:" , per  )       
        
        return r_w1 , r_w2 , r_b1 , r_b2 
               
w11 , w12 , b11 , b12 = directLearnMain()        
w21, w22, b21 , b22 = tensorflowMain()
w1diff = w11-w21
w2diff = w12-w22
d1s = np.sum( abs(w1diff) )
d2s = np.sum( abs(w2diff) )
print( "w1 diff sum" , d1s )
print( "w2 diff sum" , d2s )
b1diff = b11-b21
b2diff = b12 -b22
d1s = np.sum( abs(b1diff ) )
d2s = np.sum(abs(b2diff))
print( "b1 diff sum" , d1s )
print( "b2 diff sum" , d2s )

"""
w1:
 [[ 0.71        0.56499994  0.27499998]
 [ 1.532       0.29799998 -0.17000008]]
 """

"""
X:
[[ 0.83494319  0.11482951]
 [ 0.66899751  0.46594987]
 [ 0.60181666  0.58838408]
 [ 0.31836656  0.20502072]
 [ 0.87043944  0.02679395]
 [ 0.41539811  0.43938369]
 [ 0.68635684  0.24833404]
 [ 0.97315228  0.68541849]
 [ 0.03081617  0.89479913]
 [ 0.24665715  0.28584862]
 [ 0.31375667  0.47718349]
 [ 0.56689254  0.77079148]
 [ 0.7321604   0.35828963]
 [ 0.15724842  0.94294584]
 [ 0.34933722  0.84634483]
 [ 0.50304053  0.81299619]
 [ 0.23869886  0.9895604 ]
 [ 0.4636501   0.32531094]
 [ 0.36510487  0.97365522]
 [ 0.73350238  0.83833013]
 [ 0.61810158  0.12580353]
 [ 0.59274817  0.18779828]
 [ 0.87150299  0.34679501]
 [ 0.25883219  0.50002932]
 [ 0.75690948  0.83429824]
 [ 0.29316649  0.05646578]
 [ 0.10409134  0.88235166]
 [ 0.06727785  0.57784761]
 [ 0.38492705  0.48384792]
 [ 0.69234428  0.19687348]
 [ 0.42783492  0.73416985]
 [ 0.09696069  0.04883936]]
Y_:
[[1], [0], [0], [1], [1], [1], [1], [0], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1], [1], [1], [1], [1], [0], [1]]
w1:
[[-0.81131822  1.48459876  0.06532937]
 [-2.4427042   0.0992484   0.59122431]]
w2:
[[-0.81131822]
 [ 1.48459876]
 [ 0.06532937]]


After 0 training step(s), loss_mse on all data is 5.13118
After 500 training step(s), loss_mse on all data is 0.429111
After 1000 training step(s), loss_mse on all data is 0.409789
After 1500 training step(s), loss_mse on all data is 0.399923
After 2000 training step(s), loss_mse on all data is 0.394146
After 2500 training step(s), loss_mse on all data is 0.390597


w1:
[[-0.70006633  0.9136318   0.08953571]
 [-2.3402493  -0.14641267  0.58823055]]
w2:
[[-0.06024267]
 [ 0.91956186]
 [-0.0682071 ]]
"""

