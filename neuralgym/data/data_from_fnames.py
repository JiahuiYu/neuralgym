import random
import threading
import logging
import time

import cv2
import tensorflow as tf

from . import feeding_queue_runner as queue_runner
from .dataset import Dataset
from ..ops.image_ops import np_random_crop


logger = logging.getLogger()
READER_LOCK = threading.Lock()


class DataFromFNames(Dataset):
    """Data pipeline from list of filenames:

    Args:
        fnamelists (list): a list of filenames or tuple of filenames, e.g.
            ['image_001.png', ...] or
            [('pair_image_001_0.png', 'pair_image_001_1.png'), ...]
        shapes ():
        random (bool): read from `fnamelists` randomly (default to False)
        random_crop (bool): todo
        dtypes (tf.Type): data types, default to tf.float32
        enqueue_size (int): enqueue size for pipeline
        enqueue_size (int): enqueue size for pipeline
        nthreads (int): parallel threads for reading from data

    Examples:
        >>> # TODO
    """

    def __init__(self, fnamelists, shapes, random=False, random_crop=False,
                 fn_preprocess=None, dtypes=tf.float32,
                 enqueue_size=32, queue_size=256, nthreads=16,
                 return_fnames=False, filetype='image'):
        """ Initialize dataset. """
        self.fnamelists_ = self.process_fnamelists(fnamelists)
        self.file_length = len(self.fnamelists_)
        self.random = random
        self.random_crop = random_crop
        self.filetype = filetype
        if isinstance(shapes[0], list):
            self.shapes = shapes
        else:
            self.shapes = [shapes] * len(self.fnamelists_[0])
        if isinstance(dtypes, list):
            self.dtypes = dtypes
        else:
            self.dtypes = [dtypes] * len(self.fnamelists_[0])
        self.return_fnames = return_fnames
        self.batch_phs = [
            tf.placeholder(dtype, [None] + shape)
            for dtype, shape in zip(self.dtypes, self.shapes)]
        if self.return_fnames:
            self.shapes += [[]]
            self.dtypes += [tf.string]
            self.batch_phs.append(tf.placeholder(tf.string, [None]))
        self.enqueue_size = enqueue_size
        self.queue_size = queue_size
        self.nthreads = nthreads
        self.fn_preprocess = fn_preprocess
        if not random:
            self.index = 0
        super().__init__()
        self.create_queue()

    def process_fnamelists(self, fnamelist):
        """process fnamelist"""
        if isinstance(fnamelist, list):
            if isinstance(fnamelist[0], str):
                return [(i,) for i in fnamelist]
            elif isinstance(fnamelist[0], tuple):
                return fnamelist
            else:
                raise ValueError('Type error for fnamelist.')
        else:
            raise ValueError('Type error for fnamelist.')

    def data_pipeline(self, batch_size):
        """Return batch data queue"""
        return self._queue.dequeue_many(batch_size)

    def create_queue(self, shared_name=None, name=None):
        """Create a queue."""
        from tensorflow.python.ops import data_flow_ops, logging_ops, math_ops
        from tensorflow.python.framework import dtypes
        assert self.dtypes != None and self.shapes != None
        assert len(self.dtypes) == len(self.shapes)
        capacity = self.queue_size
        self._queue = data_flow_ops.FIFOQueue(
            capacity=capacity,
            dtypes=self.dtypes,
            shapes=self.shapes,
            shared_name=shared_name,
            name=name)

        enq = self._queue.enqueue_many(self.batch_phs)
        # create a queue runner
        queue_runner.add_queue_runner(queue_runner.QueueRunner(
            self._queue, [enq]*self.nthreads,
            feed_dict_op=[lambda: self.next_batch()],
            feed_dict_key=self.batch_phs))
        summary_name = 'fraction_of_%d_full' % capacity
        logging_ops.scalar_summary("queue/%s/%s" % (
            self._queue.name, summary_name), math_ops.cast(
                self._queue.size(), dtypes.float32) * (1. / capacity))

    def read_img(self, filename):
        """read img and return image, true/false"""
        img = cv2.imread(filename)
        if img is None:
            logger.info('image is None, sleep this thread for 0.1s.')
            time.sleep(0.1)
            return img, True
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        return img, False

    def next_batch(self):
        """
        get next batch of data
        """
        batch_data = []
        for _ in range(self.enqueue_size):
            error = True
            while error:
                error = False
                if random:
                    filenames = random.choice(self.fnamelists_)
                else:
                    with READER_LOCK:
                        filenames = self.fnamelists_[self.index]
                        self.index = (self.index + 1) % self.file_length
                imgs = []
                for i in range(len(filenames)):
                    img, error = self.read_img(filenames[i])
                    if self.random_crop:
                        img = np_random_crop(
                            img, tuple(self.shapes[i][:-1]), align=False)
                    else:
                        img = cv2.resize(img, tuple(self.shapes[i][:-1]))
                    imgs.append(img)
            if self.return_fnames:
                batch_data.append(imgs + list(filenames))
            else:
                batch_data.append(imgs)
        return zip(*batch_data)

    def _maybe_download_and_extract(self):
        pass


if __name__ == "__main__":
    import glob
    flist = glob.glob('/home/jhyu/data/imagenet/train/n01798484' + '/*.JPEG')
    # data = DataFromFNames(flist, [64, 64, 3], return_fnames=True)
    # image, names = data.data_pipeline(5)
    data = DataFromFNames(flist, [64, 64, 3])
    image = data.data_pipeline(5)
    sess = tf.Session(config=tf.ConfigProto())
    tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        # oi, on = sess.run([image, names])
        oi = sess.run(image)
        from IPython import embed; embed()
