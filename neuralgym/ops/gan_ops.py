import tensorflow as tf

from .summary_ops import scalar_summary


def gan_log_loss(pos, neg, name='gan_log_loss'):
    """
    log loss function for GANs.
    - Generative Adversarial Networks: https://arxiv.org/abs/1406.2661
    """
    with tf.variable_scope(name):
        # generative model G
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.ones_like(neg)))
        # discriminative model D
        d_loss_pos = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=pos, labels=tf.ones_like(pos)))
        d_loss_neg = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=neg, labels=tf.zeros_like(neg)))
        pos_acc = tf.reduce_mean(tf.sigmoid(pos))
        neg_acc = tf.reduce_mean(tf.sigmoid(neg))
        scalar_summary('d_scores/pos_mean', pos_acc)
        scalar_summary('d_scores/neg_mean', neg_acc)
        # loss
        d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg)
        scalar_summary('losses/d_loss', d_loss)
        scalar_summary('losses/g_loss', g_loss)
    return g_loss, d_loss


def gan_ls_loss(pos, neg, value=1., name='gan_ls_loss'):
    """
    gan with least-square loss
    """
    with tf.variable_scope(name):
        l2_pos = tf.reduce_mean(tf.squared_difference(pos, value))
        l2_neg = tf.reduce_mean(tf.square(neg))
        scalar_summary('pos_l2_avg', l2_pos)
        scalar_summary('neg_l2_avg', l2_neg)
        d_loss = tf.add(.5 * l2_pos, .5 * l2_neg)
        g_loss = tf.reduce_mean(tf.squared_difference(neg, value))
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
    return g_loss, d_loss


def gan_wgan_loss(pos, neg, name='gan_wgan_loss'):
    """
    wgan loss function for GANs.

    - Wasserstein GAN: https://arxiv.org/abs/1701.07875
    """
    with tf.variable_scope(name):
        d_loss = tf.reduce_mean(neg-pos)
        g_loss = -tf.reduce_mean(neg)
        scalar_summary('d_loss', d_loss)
        scalar_summary('g_loss', g_loss)
        scalar_summary('pos_value_avg', tf.reduce_mean(pos))
        scalar_summary('neg_value_avg', tf.reduce_mean(neg))
    return g_loss, d_loss


def random_interpolates(x, y, alpha=None):
    """
    x: first dimension as batch_size
    y: first dimension as batch_size
    alpha: [BATCH_SIZE, 1]
    """
    shape = x.get_shape().as_list()
    x = tf.reshape(x, [shape[0], -1])
    y = tf.reshape(y, [shape[0], -1])
    if alpha is None:
        alpha = tf.random_uniform(shape=[shape[0], 1])
    interpolates = x + alpha*(y - x)
    return tf.reshape(interpolates, shape)


def gradients_penalty(x, y, mask=None, norm=1.):
    """Improved Training of Wasserstein GANs

    - https://arxiv.org/abs/1704.00028
    """
    gradients = tf.gradients(y, x)[0]
    if mask is None:
        mask = tf.ones_like(gradients)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
    return tf.reduce_mean(tf.square(slopes - norm))
