# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen - extended by Ismail Elezi
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from main.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time

class RoIDataLayer(object):
  """Fast R-CNN data layer used for training."""

  def __init__(self, roidb, num_classes, random=False, augmentation='none'):
    """Set the roidb to be used by this layer during training."""
    self._roidb = roidb
    self._num_classes = num_classes
    # Also set a random flag
    self._random = random
    self._shuffle_roidb_inds()
    self.augmentation = augmentation

  def _shuffle_roidb_inds(self):
    """Randomly permute the training roidb."""
    # If the random flag is set, 
    # then the database is shuffled according to system time
    # Useful for the validation set
    if self._random:
      st0 = np.random.get_state()
      millis = int(round(time.time() * 1000)) % 4294967295
      np.random.seed(millis)

    self._perm = np.random.permutation(np.arange(len(self._roidb)))
    # Restore the random state
    if self._random:
      np.random.set_state(st0)
      
    self._cur = 0

  def _get_next_minibatch_inds(self,batch_size):
    """Return the roidb indices for the next minibatch."""
    
    if self._cur + batch_size >= len(self._roidb):
      self._shuffle_roidb_inds()

    db_inds = self._perm[self._cur:self._cur + batch_size]
    self._cur += batch_size

    return db_inds

  def _get_next_minibatch(self, args, assign, helper,lock):
    """Return the blobs to be used for the next minibatch.

    If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
    separate process and made available through self._blob_queue.
    """
    batch_size = args.batch_size
    # sync index acquiration
    if lock is not None:
      lock.acquire()
    db_inds = self._get_next_minibatch_inds(batch_size)
    if lock is not None:
      lock.release()
    minibatch_db = [self._roidb[i] for i in db_inds]
    return get_minibatch(minibatch_db, args, assign, helper, augmentation_type=self.augmentation)
      
  def forward(self, args, assign, helper=None,lock=None):
    """Get blobs and copy them into this layer's top blob vector."""
    blobs = self._get_next_minibatch(args, assign, helper,lock)
    return blobs
