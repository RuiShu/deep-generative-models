import tensorflow as tf
import tensorbayes as tb
from extra_layers import basic_accuracy, gumbel_softmax
from tensorbayes.layers import placeholder, constant
from tensorbayes.distributions import log_normal
from codebase.args import args
from pprint import pprint
exec "from designs import {:s} as des".format(args.design)
sigmoid_xent = tf.nn.sigmoid_cross_entropy_with_logits
softmax_xent = tf.nn.softmax_cross_entropy_with_logits
import numpy as np

def t2s(x):
    """
    Convert 'tanh' encoding to 'sigmoid' encoding
    """
    return (x + 1) / 2

def generate_img():
    ncol = 20
    z = np.tile(np.random.randn(ncol, 100), (10, 1))
    y = np.tile(np.eye(10), (ncol, 1))
    y = y.T.reshape(ncol * 10, -1)

    z, y = constant(z), constant(y)
    img = des.generator(z, y, phase=False, reuse=True)
    img = tf.reshape(img, [10, ncol, 32, 32, 3])
    img = tf.reshape(tf.transpose(img, [0, 2, 1, 3, 4]), [1, 10 * 32, ncol * 32, 3])
    img = t2s(img)
    return img

def vae():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    T = tb.utils.TensorDict(dict(
        sess = tf.Session(config=config),
        src_x = placeholder((None, 32, 32, 3),  name='source_x'),
        src_y = placeholder((None, 10),         name='source_y'),
        trg_x = placeholder((None, 32, 32, 3),  name='target_x'),
        trg_y = placeholder((None, 10),         name='target_y'),
        test_x = placeholder((None, 32, 32, 3), name='test_x'),
        test_y = placeholder((None, 10),        name='test_y'),
        fake_z = placeholder((None, 100),       name='fake_z'),
        fake_y = placeholder((None, 10),        name='fake_y'),
        tau = placeholder((),                   name='tau'),
        phase = placeholder((), tf.bool,        name='phase'),
    ))

    if args.gw > 0:
        # Variational inference
        y_logit = des.classifier(T.trg_x, T.phase, internal_update=True)
        y = gumbel_softmax(y_logit, T.tau)
        z, z_post = des.encoder(T.trg_x, y, T.phase, internal_update=True)

        # Generation
        x = des.generator(z, y, T.phase, internal_update=True)

        # Loss
        z_prior = (0., 1.)
        kl_z = tf.reduce_mean(log_normal(z, *z_post) - log_normal(z, *z_prior))

        y_q = tf.nn.softmax(y_logit)
        log_y_q = tf.nn.log_softmax(y_logit)
        kl_y = tf.reduce_mean(tf.reduce_sum(y_q * (log_y_q - tf.log(0.1)), axis=1))

        loss_kl = kl_z + kl_y
        loss_rec = args.rw * tf.reduce_mean(tf.reduce_sum(tf.square(T.trg_x - x), axis=[1,2,3]))
        loss_gen = loss_rec + loss_kl
        trg_acc = basic_accuracy(T.trg_y, y_logit)

    else:
        loss_kl = constant(0)
        loss_rec = constant(0)
        loss_gen = constant(0)
        trg_acc = constant(0)

    # Posterior regularization (labeled classification)
    src_y = des.classifier(T.src_x, T.phase, reuse=True)
    loss_class = tf.reduce_mean(softmax_xent(labels=T.src_y, logits=src_y))
    src_acc = basic_accuracy(T.src_y, src_y)

    # Evaluation (classification)
    test_y = des.classifier(T.test_x, phase=False, reuse=True)
    test_acc = basic_accuracy(T.test_y, test_y)
    fn_test_acc = tb.function(T.sess, [T.test_x, T.test_y], test_acc)

    # Evaluation (generation)
    if args.gw > 0:
        fake_x = des.generator(T.fake_z, T.fake_y, phase=False, reuse=True)
        fn_fake_x = tb.function(T.sess, [T.fake_z, T.fake_y], fake_x)

    # Optimizer
    var_main = tf.get_collection('trainable_variables', 'gen/')
    var_main += tf.get_collection('trainable_variables', 'enc/')
    var_main += tf.get_collection('trainable_variables', 'class/')
    loss_main = args.gw * loss_gen + loss_class
    train_main = tf.train.AdamOptimizer(args.lr, 0.5).minimize(loss_main, var_list=var_main)

    # Summarizations
    summary_main = [
        tf.summary.scalar('gen/loss_gen', loss_gen),
        tf.summary.scalar('gen/loss_rec', loss_rec),
        tf.summary.scalar('gen/loss_kl', loss_kl),
        tf.summary.scalar('class/loss_class', loss_class),
        tf.summary.scalar('acc/src_acc', src_acc),
        tf.summary.scalar('acc/trg_acc', trg_acc),
    ]
    summary_main = tf.summary.merge(summary_main)

    if args.gw > 0:
        summary_image = tf.summary.image('image/gen', generate_img())

    # Saved ops
    c = tf.constant
    T.ops_print = [
        c('tau'), tf.identity(T.tau),
        c('gen'), loss_gen,
        c('rec'), loss_rec,
        c('kl'), loss_kl,
        c('class'), loss_class,
    ]

    T.ops_main = [summary_main, train_main]
    T.fn_test_acc = fn_test_acc
    T.fn_fake_x = fn_fake_x

    if args.gw > 0:
        T.ops_image = summary_image

    return T
