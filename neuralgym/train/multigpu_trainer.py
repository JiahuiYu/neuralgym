"""Class for multi-GPU trainer."""
import time
import logging
import threading

import numpy as np
import tensorflow as tf

from tensorgym.utils.logger import ProgressBar
from tensorgym.callbacks.callbacks import *
from tensorgym.train.trainer import Trainer


logger = logging.getLogger(__name__)


class MultiGPUTrainer(Trainer):
    """Trainer class for train iterative algorithm on multi GPUs."""

    def __init__(self, **context):
        self.context = context
        self.context['graph_def'] = context.pop('graph_def')
        self.context['graph_def_kwargs'] = context.pop('graph_def_kwargs')
        self.context['async_train'] = context.pop('async_train', False)
        self.train_op, self.context['loss'] = self.train_ops_and_losses()
        if self.context['async_train']:
            self.context['train_op'] = self.train_op[0]
        else:
            self.context['train_op'] = self.train_op
        super().__init__(**self.context)

    def train(self):
        """start training with callbacks"""
        def train_function(sess, train_op):
            """"""
            while True:
                sess.run(train_op)

        if not self.context['async_train']:
            super().train()
        else:
            train_threads = []
            for i, train_op in enumerate(self.train_op):
                if i == 0:
                    # main thread
                    pass
                else:
                    train_threads.append(
                        threading.Thread(
                            target=train_function, args=(
                                self.context['sess'], train_op,)))
            # Start the threads, and block on their completion.
            try:
                for t in train_threads:
                    logger.info("Start new thread for async training.")
                    t.start()
                # start main thread
                super().train()
                for t in train_threads:
                    t.join()
            except (KeyboardInterrupt, SystemExit):
                logger.info("Training is stoped.")

    def average_gradients(self, tower_grads):
        """ Calculate the average gradient for each shared variable across
        all towers.

        **Note** that this function provides a synchronization point
        across all towers.

        :param tower_grads: List of lists of (gradient, variable) tuples.
            The outer list is over individual gradients. The inner list is
            over the gradient calculation for each tower.
        :returns: List of pairs of (gradient, variable) where the gradient
            has been averaged across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            v = grad_and_vars[0][1]
            # sum
            grad = tf.add_n([x[0] for x in grad_and_vars])
            # average
            grad = grad / float(len(tower_grads))
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train_ops_and_losses(self):
        """Return loss of towers on different gpus.
        """
        optimizer = self.context['optimizer']
        var_list = self.context.get('var_list')
        self.context['global_step'] = self.context.pop(
            'global_step', tf.get_variable(
                'global_step', [], dtype=tf.int32,
                initializer=tf.zeros_initializer(), trainable=False))
        global_step = self.context['global_step']
        graph_def_kwargs = self.context['graph_def_kwargs']
        gradient_processor = self.context.get('gradient_processor')
        tower_grads = []
        tower_losses = []
        for gpu in range(self.context['gpu_num']):
            with tf.device('/gpu:%d' % gpu):
                # with tf.name_scope('tower_gpu%d' % gpu) as scope:
                # Reuse variables for the next tower.
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    loss = self.context['graph_def'](gpu, **graph_def_kwargs)
                    tower_losses.append(loss)
                    # Calculate the gradients for the batch of data
                    grads = optimizer.compute_gradients(loss, var_list)
                    if self.context['grads_summary']:
                        for grad, var in grads:
                            if grad is not None:
                                tf.summary.histogram(
                                    'gradients/' + var.name, grad)
                    # process gradients
                    if gradient_processor is not None:
                        grads = [grad for grad in grads if grad[0] is not None]
                        grads = [gradient_processor(grad) for grad in grads]
                    # Keep track of the gradients across all towers.
                    tower_grads.append(grads)
        if self.context['async_train']:
            apply_gradient_op = []
            loss = tower_losses[0]
            for i in range(len(tower_grads)):
                if i != 0:
                    global_step = None
                apply_gradient_op.append(
                    optimizer.apply_gradients(
                        tower_grads[i], global_step=global_step))
        else:
            # average gradients.
            grads = self.average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(
                grads, global_step=global_step)
            loss = tf.reduce_mean(tower_losses)
        return apply_gradient_op, loss
