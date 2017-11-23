""" logger utilities """
import logging
import shutil
import sys
import time
from termcolor import colored


logger = logging.getLogger(__name__)


def colored_log(prompt, texts, color='green', attrs=None):
    """Show colored logs.
    """
    assert isinstance(prompt, str)
    assert isinstance(texts, str)
    assert isinstance(color, str)
    if attrs is None:
        attrs = ['bold']
    colored_prompt = colored(prompt, color, attrs=attrs)
    clean_line = '\x1b[2K\r'
    sys.stdout.write(clean_line)
    logger.info(colored_prompt + texts)


def callback_log(texts):
    """Callback_log will show caller's location.
    """
    colored_log('Trigger callback: ', texts)


def warning_log(texts):
    """Warning_log will show caller's location and red texts.
    """
    colored_log('Warning: ', texts, color='red')


def error_log(texts):
    """Error_log will show caller's location, red texts and raise
    RuntimeError.
    """
    colored_log('Error: ', texts, color='red')
    raise RuntimeError


class ProgressBar(object):
    """ Visualize progress.

    It displays a progress bar in console with time recorder and statistics.
    """

    def __init__(self):
        # length of progress bar = terminal width / split_n
        self._split_n = 4
        self._t_start = None
        self._t_last = None
        self._t_current = None
        # progress recorders
        self._p_start = None
        self._p_last = None
        self._p_current = None
        # restart at init
        self.restart()

    def restart(self):
        """Restart time recorder and progress recorder."""
        # time recorders
        self._t_start = time.time()
        self._t_last = self._t_start
        self._t_current = self._t_start
        # progress recorders
        self._p_start = 0
        self._p_last = self._p_start
        self._p_current = self._p_start

    def progress(self, progress, texts=''):
        """Update progress bar with current progress and additional texts.

        :param progress: float between [0,1] indicating progress
        :param texts: additional texts (e.g. statistics) appear at the end
            of progress bar
        """
        term_length, _ = shutil.get_terminal_size()
        length = int(term_length / self._split_n)
        if isinstance(progress, int):
            progress = float(progress)
        assert isinstance(progress, float)
        assert isinstance(texts, str)
        assert progress >= 0 and progress <= 1, 'Progress is between [0,1].'
        # the number of '#' to be shown
        block = int(round(length*progress))
        # compute time and progress
        self._p_current = progress
        self._t_current = time.time()
        speed = (self._p_current-self._p_last)/(self._t_current-self._t_last)
        t_consumed = self._t_current - self._t_start
        t_remained = (1-self._p_current) / speed if speed != 0 else 0
        # info to be shown
        info = ''.join([
            '\x1b[2K\r|',
            '#'*block,
            '-'*(length-block),
            '|',
            ' {:.2f}%,'.format(self._p_current*100),
            ' {:.0f}/{:.0f} sec.'.format(t_consumed, t_remained),
            ' {}'.format(texts)
        ])

        if progress == 1:
            logger.info(info)
            self.restart()
        else:
            sys.stdout.write(info)
            # update time and progress
            self._p_last = self._p_current
            self._t_last = self._t_current
