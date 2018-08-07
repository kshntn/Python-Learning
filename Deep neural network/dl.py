import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-true')

fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name='hidden-layer')
y = tf.layers.dense(inputs=fc1, units=10, name='output')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('train', sess.graph)


def feed_dict(is_training):
    if is_training:
        batch_x, batch_y = mnist.train.next_batch(100)
    else:
        batch_x, batch_y = mnist.test.images, mnist.test.labels
    return {x: batch_x, y_: batch_y}


for i in range(3000):
    if i % 100 == 0:
        # saver.save(sess, os.path.join(CHECKPOINTS_FOLDER, 'model'), global_step=i)
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
        print 'Step %d: training accuracy: %f' % (i, acc)
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
        train_writer.flush()

train_writer.close()

print 'Test accuracy: ', sess.run(accuracy, feed_dict=feed_dict(False))
