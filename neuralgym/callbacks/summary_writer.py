"""summary writer"""
from .callbacks import PeriodicCallback, CallbackLoc


class SummaryWriter(PeriodicCallback):
    """Periodically add summary.

    :param pstep: call summary writer every pstep
    :param summary_writer: tensorflow summary writer
    :param summary: tensorflow summary collection
    """

    def __init__(self, pstep, summary_writer, summary):
        super().__init__(CallbackLoc.step_end, pstep)
        self._summary_writer = summary_writer
        self._summary = summary

    def run(self, sess, step):
        # ignore callback_log for summary writer
        self._summary_writer.add_summary(sess.run(self._summary), step)
