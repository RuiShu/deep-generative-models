from codebase.models.extra_layers import leaky_relu
import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorbayes.layers import dense, conv2d, conv2d_transpose, avg_pool, max_pool, batch_norm, gaussian_sample
from codebase.args import args

dropout = tf.layers.dropout

def classifier(x, phase, scope='class', reuse=None, internal_update=False, getter=None):
    with tf.variable_scope(scope, reuse=reuse, custom_getter=getter):
        with arg_scope([leaky_relu], a=0.1), \
             arg_scope([conv2d, dense], activation=leaky_relu, bn=True, phase=phase), \
             arg_scope([batch_norm], internal_update=internal_update):

            layout = [
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (conv2d, (96, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (max_pool, (2, 2), {}),
                (dropout, (), dict(training=phase)),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (conv2d, (192, 3, 1), {}),
                (avg_pool, (), dict(global_pool=True)),
                (dense, (10,), dict(activation=None))
            ]

            start = 0
            end = len(layout)

            for i in xrange(start, end):
                with tf.variable_scope('l{:d}'.format(i)):
                    f, f_args, f_kwargs = layout[i]
                    x = f(x, *f_args, **f_kwargs)

    return x

def encoder(x, y, phase, scope='enc', reuse=None, internal_update=False):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([conv2d, dense], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([batch_norm], internal_update=internal_update):

            # Ignore y
            x = conv2d(x, 64, 3, 2)
            x = conv2d(x, 128, 3, 2)
            x = conv2d(x, 256, 3, 2)
            x = dense(x, 1024)

            # Autoregression (4 steps)
            ms = []
            vs = []
            zs = [x]

            for i in xrange(5):
                h = tf.concat(zs, axis=-1)
                h = dense(h, 100)
                m = dense(h, 20, activation=None)
                v = dense(h, 20, activation=tf.nn.softplus) + 1e-5
                z = gaussian_sample(m, v)
                ms += [m]
                vs += [v]
                zs += [z]

            m = tf.concat(ms, 1)
            v = tf.concat(vs, 1)
            z = tf.concat(zs[1:], 1)

    return z, (m, v)

def generator(z, y, phase, scope='gen', reuse=None, internal_update=False):
    with tf.variable_scope(scope, reuse=reuse):
        with arg_scope([dense, conv2d_transpose], bn=True, phase=phase, activation=leaky_relu), \
             arg_scope([batch_norm], internal_update=internal_update):

            x = tf.concat([z, y], 1)
            x = dense(x, 4 * 4 * 512)
            x = tf.reshape(x, [-1, 4, 4, 512])
            x = conv2d_transpose(x, 256, 5, 2)
            x = conv2d_transpose(x, 128, 5, 2)
            x = conv2d_transpose(x, 3, 5, 2, bn=False, activation=tf.nn.tanh)

    return x
