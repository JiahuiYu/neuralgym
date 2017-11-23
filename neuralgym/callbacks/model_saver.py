"""model_saver"""
import os

from .callbacks import PeriodicCallback, CallbackLoc
from ..utils.logger import callback_log


class ModelSaver(PeriodicCallback):
    """Save model to file at every pstep step_start.

    :param pstep: save to model every pstep
    :param saver: tensorflow saver
    :param dump_prefix: prefix for saving model files
    """

    def __init__(self, pstep, saver, dump_prefix):
        super().__init__(CallbackLoc.step_start, pstep)
        self._saver = saver
        self._dump_prefix = dump_prefix
        dump_dir = os.path.dirname(self._dump_prefix)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
            callback_log('Initialize ModelSaver: mkdirs %s.' % dump_dir)

    def run(self, sess, step):
        if step != 0:
            callback_log('Trigger ModelSaver: Save model to {}-{}.'.format(
                self._dump_prefix, step))
            self._saver.save(sess, self._dump_prefix, global_step=step)
