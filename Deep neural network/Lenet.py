import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-true')
    keep_prob = tf.placeholder(tf.float32)
    x_img = tf.reshape(x, [-1, 28, 28, 1])

'''
fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name='hidden-layer')
y = tf.layers.dense(inputs=fc1, units=10, name='output')
'''


def weight_var(shape):
    return tf.get_variable('weights', shape=shape, initializer=tf.contrib.layers.xavier_initializer_conv2d())


def bias_var(shape):
    return tf.get_variable('bias', shape=shape, initializer=tf.constant_initializer(0.1))


def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.variable_scope('conv1'):
    W_conv1 = weight_var([5, 5, 1, 32])
    b_conv1 = bias_var([32])
    preact_conv1 = conv(x_img, W_conv1) + b_conv1
    act_conv1 = tf.nn.relu(preact_conv1)
    # 28x28x32
    with tf.variable_scope('viz'):
        W_min=tf.reduce_min(W_conv1)
        W_max=tf.reduce_max(W_conv1)
        W_norm=(W_conv1-W_min)/(W_max-W_min)

        W_transpose=tf.transpose(W_norm,[3,0,1,2])
        tf.summary.image('filters',W_transpose,max_outputs=10)

with tf.variable_scope('pool1'):
    pool1 = max_pool_2x2(act_conv1)
#     14x14x32

with tf.variable_scope('conv2'):
    W_conv2 = weight_var([5, 5, 32, 64])
    b_conv2 = bias_var([64])
    preact_conv2 = conv(pool1, W_conv2) + b_conv2
    act_conv2 = tf.nn.relu(preact_conv2)

with tf.variable_scope('pool2'):
    pool2 = max_pool_2x2(act_conv2)

with tf.variable_scope('fc1'):
    pool2_flat = tf.contrib.layers.flatten(pool2)
    flat_dim = pool2_flat.get_shape()[1]

    W_fc1 = weight_var([flat_dim, 1024])
    b_fc1 = bias_var([1024])
    preact_fc1 = tf.matmul(pool2_flat, W_fc1) + b_fc1
    act_fc1 = tf.nn.relu(preact_fc1)
    fc1_dropout = tf.nn.dropout(act_fc1, keep_prob=keep_prob)

with tf.variable_scope('fc2'):
    W_fc2 = weight_var([1024, 10])
    b_fc2 = bias_var([10])
    y = tf.matmul(fc1_dropout, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
tf.summary.scalar('cross_entropy', loss)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('train', sess.graph)


def feed_dict(is_training):
    if is_training:
        batch_x, batch_y = mnist.train.next_batch(1000)
        prob = 0.5
    else:
        batch_x, batch_y = mnist.test.images, mnist.test.labels
        prob = 1.0
    return {x: batch_x, y_: batch_y, keep_prob: prob}


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
