#coding:utf-8
import tensorflow as tf
import numpy as np

BATCH_SIZE = 2
SAMPLE_SIZE = 6

def  matrixCopy( m , mNew ) :
    for  r  in   range( len( m ) )  :
        for c in  range ( len ( m[r] ) ) :
            mNew[r][c] = m[r][c]


def forwardnn(xa, w1 , w2 , ya) :
    i = 0 
    for  x in xa :
        a1 = np.dot( x , w1 )
        y = np.dot( a1 , w2 ) 
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


def  learn( layer ,  w1, w2 , wnew ,  xa , ya , y_a , learnRate , wDelta  ) :
    forwardnn( xa ,  w1 , w2 , ya )
    e1 = squareErrorAvg( y_a , ya )
    if layer == 1 :
        w = w1
    else:
        w = w2 
    for  r  in   range( len( wnew ) )  :
        for c in  range ( len ( wnew[r] ) ) :
            v = w[r][c]
            w[r][c] += wDelta
            forwardnn( xa ,  w1 , w2 , ya )
            e2 = squareErrorAvg( y_a , ya )
            learnN = (e2-e1)/wDelta * learnRate # (e2-e1)/wDelta :loss函數的微分 
            w[r][c] =  v
            wnew[r][c]  =  v- learnN

''' Main for  Leaned by myself '''
def learnMain() :
    X = [
        [1.0,2.0] , [2.0,3.0]  ,[1.0,4.0],[2.0,5.0] , [1.0,1.0] , [1.5,1.0]  ]
    Y_ = [[0.0], [1.0] ,[0.0], [0.0] ,[1.0], [1.0]] 

    w1= np.array(  [[1.0,1.0,1.0],[2.0,1.0,1.0]] ) 
    w2= np.array( [[2.0],[3.0],[5.0]] ) 

    Y = [    [0.0], [0.0] ,[0.0], [0.0] ,[0.0], [0.0] ]  
       
    STEPS = 3000
    w1new = [[0.0,0.0,0.0],[0.0,0.0,0.0]]
    w2new= [[0.0],[0.0],[0.0]]
    for i in range(STEPS):
        start =  (i*BATCH_SIZE) % SAMPLE_SIZE 
        end =  start + BATCH_SIZE
        xa = X[start:end]
        ya = Y[start:end]
        y_a = Y_[start:end]
        learn( 1 ,  w1, w2 , w1new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        learn( 2 ,  w1, w2 , w2new ,  xa , ya , y_a , 0.001 ,0.0000001 )
        matrixCopy( w1new ,    w1)
        matrixCopy( w2new ,    w2)
        if  (i +1)% 10000  == 0 :
            forwardnn( xa ,  w1 , w2 , ya )
            e = squareErrorAvg( y_a  , ya )
            print ('w1 learned  by myself @time(s)  ' , i+1 , '  : ' ,  w1new )
            print ('w2 learned  by myself @time(s)  ' ,  i +1, '  : '  ,  w2new )
            print('loss  by myself @time(s)  ' ,  i +1, '  : '   , e) 
    print( "训练结果(myself)：" )
    print("w1:", w1 )
    print("w2:", w2 )
    return w1,w2


''' main for training using tensorflow '''
def main() :
    
    X = [
        [1.0,2.0] , [2.0,3.0]  ,[1.0,4.0],[2.0,5.0] , [1.0,1.0] , [1.5,1.0]  ]
    Y_ = [[0.0], [1.0] ,[0.0], [0.0] ,[1.0], [1.0]] 

    #1定义神经网络的输入、参数和输出,定义前向传播过程。
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_= tf.placeholder(tf.float32, shape=(None, 1))

    w1= tf.Variable( [[1.0,1.0,1.0],[2.0,1.0,1.0]])
    w2= tf.Variable([[2.0],[3.0],[5.0]])

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    #2定义损失函数及反向传播方法。
    loss_mse = tf.reduce_mean(tf.square(y-y_)) 
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)
    #3生成会话，训练STEPS轮
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
          
        # 训练模型。
        STEPS = 3000
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
        print("w1:", r_w1  )
        print("w2:", r_w2 )
        return r_w1 , r_w2 
               
w11 , w12 = learnMain()        
w21, w22 = main()
w1diff = w11-w21
w2diff = w12-w22
#print ( "w11,w12" , w11, w12 )
#print ( "w21,w22" , w21, w22 )
#print("diff" , w1diff , w2diff )
d1s = np.sum( w1diff )
d2s = np.sum( w2diff )
print( "w1 diff sum" , d1s )
print( "w2 diff sum" , d2s )

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

