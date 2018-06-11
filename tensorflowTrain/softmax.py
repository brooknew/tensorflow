
import tensorflow as tf
import math

#a = [1.0,2.0,3.0,4.0]
a = tf.Variable( [1.0,2.0,3.0,4.0] )
y = tf.nn.softmax( a )

se = tf.Session()
init = tf.global_variables_initializer()
se.run( init ) 
r = se.run( y )
print( y )
print( r )

print( r[0] +  r[1] + r[2] + r[3] )
#print( a[0] ) 
all =  math.exp( 1.0) + math.exp( 2.0) + math.exp( 3.0) + math.exp( 4.0)
print("all:" , all )
print(  math.exp( 1.0) /all)
print(  math.exp( 2.0)/all )
print(  math.exp( 3.0)/all )
print(  math.exp( 4.0) /all)

