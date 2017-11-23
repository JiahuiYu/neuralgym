"""NeuralGym"""
import datetime
import logging
import os

from logging.config import dictConfig
from termcolor import colored

from . import callbacks
from . import ops
from . import train
from .utils.gpus import set_gpus, get_gpus
from .utils.config import Config



__all__ = ['date_uid', 'get_gpus', 'set_gpus', 'Config']


def date_uid():
    """generate a unique id based on date

    :returns: string of number, e.g. 20171122171307111552
    """
    return str(datetime.datetime.now()).replace(
        '-', '').replace(
            ' ', '').replace(
                ':', '').replace('.', '')


LOGGER_DIR = 'neuralgym_logs/' + str(date_uid())
LOGGER_FILE = 'neuralgym.log'
LOGGING_CONFIG = dict(
    version=1,
    formatters={
        'f': {
            'format': colored('\x1b[2K\r[%(asctime)s @%(filename)s:%(lineno)d'
                              ']green') + ' %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'},
        'ff': {
            'format':
            '[\x1b[2K\r%(levelname)-8s %(asctime)s @%(filename)s:%(lineno)d] '
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
