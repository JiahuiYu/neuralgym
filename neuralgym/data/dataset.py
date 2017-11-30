from abc import abstractmethod
import logging

import tensorflow as tf


logger = logging.getLogger()


class Dataset(object):
    """Base class for datasets.

    Dataset members are automatically logged except members with name ending
    of '_', e.g. 'self.fnamelists_'.
    """

    def __init__(self):
        self.maybe_download_and_extract()
        self.view_dataset_info()

    def view_dataset_info(self):
        """Function to view current dataset information.

        """
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
    def maybe_download_and_extract(self):
        """Abstract class: dataset maybe need download items.

        """
        pass

    @abstractmethod
    def data_pipeline(self, batch_size):
        """Return batch data with batch size, e.g. return batch_image or
        return (batch_data, batch_label).

        Args:
            batch_size (int): Batch size.

        """
        pass
