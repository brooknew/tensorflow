import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape, regularizer , vname):
    w = tf.Variable(tf.truncated_normal(shape,stddev=0.1) , name = vname )
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape , vname):  
    b = tf.Variable(tf.zeros(shape) , name = vname)  
    return b
	
def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer , 'w1')
    b1 = get_bias([LAYER1_NODE] , 'b1')
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer , 'w2')
    b2 = get_bias([OUTPUT_NODE] , 'b2')
    y = tf.matmul(y1, w2) + b2
    return y
