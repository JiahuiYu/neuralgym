import tensorflow as tf


__all__ = ['get_sess']


def get_sess(sess=None):
    """Get default session if sess is None.

    :param sess: valid sess or None
    :return: valid sess
    """
    if sess is None:
        sess = tf.get_default_session()
    assert sess, 'sess be None.'
    return sess
