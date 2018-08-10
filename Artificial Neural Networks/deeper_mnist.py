import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)

with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y_')

with tf.variable_scope('layer'):
    fc1 = tf.layers.dense(inputs=X, units=512, activation=tf.nn.relu)
    fc2 = tf.layers.dense(inputs=fc1, units=512, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=fc2, units=10, activation=tf.nn.relu)

with tf.variable_scope('output'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('loss', cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.variable_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs', graph=sess.graph)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(256)
    summary, loss, _ = sess.run([merged, cross_entropy, train_step], feed_dict={X: batch_xs, y_: batch_ys})
    if (i + 1) % 100 == 0:
        print 'Iteration %d %f' % (i + 1, loss)
    train_writer.add_summary(summary, i)
    train_writer.flush()

print sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels})
train_writer.close()
