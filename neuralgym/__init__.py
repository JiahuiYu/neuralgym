import datetime
import logging
import os

from logging.config import dictConfig
from .utils.logger import colorize


__version__ = '0.0.1'
__all__ = ['Config', 'get_gpus', 'set_gpus', 'date_uid', 'unset_logger',
           'get_sess']


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return str(datetime.datetime.now()).replace(
        '-', '').replace(
            ' ', '').replace(
                ':', '').replace('.', '')


LOGGER_DIR = 'neuralgym_logs/' + date_uid()
LOGGER_FILE = 'neuralgym.log'
LOGGING_CONFIG = dict(
    version=1,
    formatters={
        'f': {
            'format': colorize('[%(asctime)s @%(filename)s:%(lineno)d]',
                               'green') + ' %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'},
        'ff': {
            'format':
            '[%(levelname)-5s %(asctime)s @%(filename)s:%(lineno)d] '
            '%(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'}
        },
    handlers={
        'h': {
            'class': 'logging.StreamHandler', 'formatter': 'f',
            'level': logging.DEBUG},
        'fh': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(LOGGER_DIR, LOGGER_FILE),
            'formatter': 'ff',
            'level': logging.DEBUG}
        },
    root={
        'handlers': ['h', 'fh'],
        'level': logging.DEBUG,
        },
)


if not os.path.exists(LOGGER_DIR):
    os.makedirs(LOGGER_DIR)


dictConfig(LOGGING_CONFIG)


from . import callbacks
from . import ops
from . import train
from . import models
from . import data
from . import server

from .utils.gpus import set_gpus, get_gpus
from .utils.tf_utils import get_sess
from .utils.config import Config


logger = logging.getLogger()
logger.info('Set root logger. Unset logger with neuralgym.unset_logger().')
logger.info('Saving logging to file: {}.'.format(LOGGER_DIR))


def unset_logger():
    """Unset logger of neuralgym.

    Todo:
        * How to unset logger properly.

    """
    raise NotImplementedError('Unset logger function is not implemented yet.')
