import tensorflow as tf   
import numpy as np

SIZE=6
CLASS=10
label=np.random.randint(0,10,size=SIZE) 
print(label)
label=np.reshape(label,[SIZE,1])
print(label)
ind =  np.arange(SIZE)
print( "arange:" , ind )
index = np.reshape( ind, [SIZE, 1])
print( index )
#use a matrix  
concated = tf.concat([index, label], 1)  
onehot_labels = tf.sparse_to_dense(concated, [SIZE, CLASS], 1.0, 0.0)  

#use a vector  
concated2=tf.constant([1,3,4])  
onehot_labels2 = tf.sparse_to_dense(concated2, [ CLASS], 1.0, 0.0)

#use a scalar  
concated3=tf.constant(5)  
onehot_labels3 = tf.sparse_to_dense(concated3, [ CLASS], 1.0, 0.0)  

with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())
    c1 =  sess.run(concated)
    print( c1 )
    result1=sess.run(onehot_labels)  
    result2 = sess.run(onehot_labels2)  
    result3 = sess.run(onehot_labels3)  
    print ("This is result1:")  
    print (result1)  
    print ("This is result2:")  
    print (result2)  
    print ("This is result3:")  
    print (result3) 
