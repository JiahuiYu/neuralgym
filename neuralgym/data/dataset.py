"""Base class for dataset"""
from abc import abstractmethod
import logging

import tensorflow as tf

logger = logging.getLogger()


class Dataset(object):
    """Base class for datasets.
    """

    def __init__(self):
        """ Initialize dataset. """
        self._maybe_download_and_extract()
        self._view_dataset_info()

    def _view_dataset_info(self):
        """ function to view current dataset information """
        dicts = vars(self)
        logger.info(' Dataset Info '.center(80, '-'))
        for key in dicts:
            # not in a recursive way.
            if isinstance(dicts[key], dict):
                logger.info('%s:', key)
                tmp_dicts = dicts[key]
                for tmp_key in tmp_dicts:
                    logger.info('  %s: %s', tmp_key, tmp_dicts[tmp_key])
            else:
                if key[-1] != '_':
                    logger.info('%s: %s', key, dicts[key])
        logger.info(''.center(80, '-'))

    @abstractmethod
    def _maybe_download_and_extract(self):
        """ abstract class: dataset maybe need download items. """
        pass

    @abstractmethod
    def data_pipeline(self, subset, batch_size):
        """Return batch data. e.g. return batch_image, batch_label"""
        pass
