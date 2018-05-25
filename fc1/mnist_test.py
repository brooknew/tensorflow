#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
TEST_INTERVAL_SECS = 5
"""
{'Variable/ExponentialMovingAverage':  <tf.Variable 'Variable:0' shape=(784, 500) dtype=float32_ref>,
 'Variable_3/ExponentialMovingAverage': <tf.Variable 'Variable_3:0' shape=(10,) dtype=float32_ref>,
 'Variable_2/ExponentialMovingAverage': <tf.Variable 'Variable_2:0' shape=(500, 10) dtype=float32_ref>,
 'Variable_1/ExponentialMovingAverage': <tf.Variable 'Variable_1:0' shape=(500,) dtype=float32_ref>
 }
"""

def test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        print( ema_restore )
        saver = tf.train.Saver(ema_restore)
		
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print ( ckpt.model_checkpoint_path)
                    print( ckpt.model_checkpoint_path.split('/')[-1] )
                    print(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS)

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)

if __name__ == '__main__':
    main()
