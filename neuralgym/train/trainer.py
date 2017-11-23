"""Class for single-GPU trainer."""
import time
import logging

import numpy as np
import tensorflow as tf

from ..utils.logger import ProgressBar
from ..callbacks.callbacks import CallbackLoc
from ..callbacks.callbacks import PeriodicCallback, OnceCallback
from ..callbacks.callbacks import ScheduledCallback


logger = logging.getLogger(__name__)


class Trainer(object):
    """Trainer class for train iterative algorithm on single GPU.

    Trainer contains all tensorflow related instances and configurations. TODO
    Trainer has objective (loss), callbacks, and all context information
    including:

    * optimizer
    * spe
    * feed_dict
    * max_iters
    * log_dir
    * global_step
    * sess_config
    * allow_growth
    * gradient_processor
    * loss
    * summary writer
    * saver
    * global_variables_initializer
    * start_queue_runners

    :param optimizer:
    :param spe:
    :param feed_dict:
    :param max_iters:
    :param log_dir:
    :param global_step:
    :param sess_config:
    :param allow_growth:
    :param gradient_processor:
    :param loss:
    :param summary_writer:
    :param saver:
    :param global_variables_initializer:
    :param start_queue_runners:
    """

    def __init__(self, **context):
        self.context = context
        self.callbacks = context.pop('callbacks', [])
        # contexts
        self.context['feed_dict'] = context.pop('feed_dict', {})
        self.context['max_iters'] = int(context.pop('max_iters', 9999999))
        self.context['log_dir'] = context.pop('log_dir', '/tmp/tensorgym/')
        self.context['sess_config'] = context.pop(
            'sess_config', tf.ConfigProto())
        self.context['sess_config'].gpu_options.allow_growth = context.pop(
            'allow_growth', True)
        self.context['sess_config'].allow_soft_placement = context.pop(
            'allow_soft_placement', True)
        self.context['sess'] = tf.Session(config=self.context['sess_config'])
        self.context['summary_writer'] = tf.summary.FileWriter(
            self.context['log_dir'], self.context['sess'].graph)
        self.context['saver'] = tf.train.Saver(tf.global_variables())
        # grads summary
        self.context['grads_summary'] = context.pop(
            'grads_summary', True)
        # train ops and losses
        self._train_op = context.pop('train_op', None)
        if self._train_op is None:
            self._train_op, self._loss = self.train_ops_and_losses()
        else:
            self._loss = context.pop('loss', 0)
        # global step
        self._bar = ProgressBar()
        # callbacks types
        self._periodic_callbacks = None
        self._once_callbacks = None
        self._scheduled_callbacks = None
        # queue runner
        self.context['start_queue_runners'] = context.pop(
            'start_queue_runner', True)
        if self.context['start_queue_runners']:
            tf.train.start_queue_runners(sess=self.context['sess'])
        # initialization
        self.context['global_variables_initializer'] = context.pop(
            'global_variables_initializer', True)
        if self.context['global_variables_initializer']:
            self.context['sess'].run(tf.global_variables_initializer())
        # total loss, beginning timepoint
        self._log_stats = [0, None]
        # log context of trainer
        logger.info(' Context Of Trainer '.center(80, '-'))
        for k in self.context:
            logger.info(k + ': ' + str(self.context[k]))
        logger.info(''.center(80, '-'))

    def train(self):
        """start training with callbacks."""
        sess = self.context['sess']
        max_iters = self.context['max_iters']
        self.update_callbacks()
        step = 0
        # once_callbacks at train start
        for cb in self._once_callbacks:
            if cb.cb_loc == CallbackLoc.train_start:
                cb.run(sess)
        try:
            while step < max_iters:
                # get current step
                step = sess.run(self.context['global_step'])
                # periodic callbacks at step start
                for cb in self._periodic_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_start and
                            step % cb.pstep == 0):
                        cb.run(sess, step)
                # scheduled callbacks at step start
                for cb in self._scheduled_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_start and
                            step in cb.schedule):
                        cb.run(sess, step)
                # run train op
                _, loss_value = sess.run([self._train_op, self._loss],
                                         feed_dict=self.context['feed_dict'])
                # if nan, exist
                assert not np.isnan(loss_value)
                # log one
                self.progress_logger(step, loss_value)
                # periodic callbacks at step end
                for cb in self._periodic_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_end and
                            step % cb.pstep == 0):
                        cb.run(sess, step)
                # scheduled callbacks at step end
                for cb in self._scheduled_callbacks:
                    if (cb.cb_loc == CallbackLoc.step_end and
                            step in cb.schedule):
                        cb.run(sess, step)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Training is stoped.")
        except:
            raise
        finally:
            # once_callbacks at exception
            for cb in self._once_callbacks:
                if cb.cb_loc == CallbackLoc.exception:
                    cb.run(sess)
        # once_callbacks at train end
        for cb in self._once_callbacks:
            if cb.cb_loc == CallbackLoc.train_end:
                cb.run(sess)

    def progress_logger(self, step, loss):
        """Progress bar for logging.

        **Note** that all statistics are averaged over epoch.
        """
        # init
        if self._log_stats[1] is None:
            self._log_stats[1] = time.time()
            self._log_stats[0] = loss
            return
        # update statistic
        self._log_stats[0] += loss
        # time
        t_start = self._log_stats[1]
        t_now = time.time()
        spe = self.context['spe']
        # after running the session, the step is actually increased.
        step = step + 1
        epoch_end = (step % spe == 0)
        # set update step 0.1%
        log_per_iters = max(int(spe/1000), 10)
        # update progress bar per log_per_iters
        epoch_nums = (step - 1) // spe + 1
        epoch_iters = (step - 1) % spe + 1
        if epoch_iters % log_per_iters == 0 or epoch_end:
            batches_per_sec = epoch_iters / (t_now - t_start)
            texts = ''.join([
                'train epoch {},'.format(epoch_nums),
                ' iter {}/{},'.format(epoch_iters, spe),
                ' loss {:.6f}, {:.2f} batches/sec.'.format(
                    self._log_stats[0]/epoch_iters, batches_per_sec),
            ])
            # progress, if at the end of epoch, 100%; else current progress
            prog = 1 if epoch_end else (step / spe) % 1
            self._bar.progress(prog, texts)
        # reset
        if epoch_end:
            self._log_stats[1] = None
            self._log_stats[0] = 0
        return

    def add_callbacks(self, callbacks):
        """ add callbacks """
        # keep order
        self.callbacks = self.callbacks + callbacks
        # after add callbacks, update callbacks list.
        self.update_callbacks()

    def update_callbacks(self):
        """update callbacks to sub-callbacks"""
        def _check_type(t, cb):
            """check callback types"""
            return t == cb.__class__ or t in cb.__class__.__bases__
        # clear
        self._periodic_callbacks = []
        self._once_callbacks = []
        self._scheduled_callbacks = []
        # add
        for cb in self.callbacks:
            if _check_type(PeriodicCallback, cb):
                self._periodic_callbacks.append(cb)
            if _check_type(OnceCallback, cb):
                self._once_callbacks.append(cb)
            if _check_type(ScheduledCallback, cb):
                self._scheduled_callbacks.append(cb)

    def train_ops_and_losses(self):
        """define train ops and losses"""
        optimizer = self.context['optimizer']
        self.context['global_step'] = self.context.pop(
            'global_step', tf.get_variable(
                'global_step', [], dtype=tf.int32,
                initializer=tf.zeros_initializer(), trainable=False))
        global_step = self.context['global_step']
        loss = self.context['loss']
        var_list = self.context.get('var_list')
        gradient_processor = self.context.get('gradient_processor')
        if loss is None:
            raise ValueError('loss is not defined.')
        # get gradients
        grads = optimizer.compute_gradients(loss, var_list)
        if self.context['grads_summary']:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram('gradients/' + var.name, grad)
        # process gradients
        if gradient_processor is not None:
            grads = [grad for grad in grads if grad[0] is not None]
            grads = [gradient_processor(grad) for grad in grads]
        # get operations
        apply_gradient_op = optimizer.apply_gradients(
            grads, global_step=global_step)
        return apply_gradient_op, loss
