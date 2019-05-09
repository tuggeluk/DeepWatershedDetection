# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
#from main.config import cfg
from main.bbox_transform import bbox_transform
from utils.bbox import bbox_overlaps
import PIL

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  for i in range(len(imdb.image_index)):
    if not (imdb.name.startswith('macrophages')):
      roidb[i]['image'] = imdb.image_path_at(i)
      if not (imdb.name.startswith('coco')):
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
      # need gt_overlaps as a dense array for argmax
      gt_overlaps = roidb[i]['gt_overlaps'].toarray()
      # max overlap with gt over classes (columns)
      max_overlaps = gt_overlaps.max(axis=1)
      # gt class that had the max overlap
      max_classes = gt_overlaps.argmax(axis=1)
      roidb[i]['max_classes'] = max_classes
      roidb[i]['max_overlaps'] = max_overlaps
      # sanity checks
      # max overlap of 0 => class should be zero (background)
      zero_inds = np.where(max_overlaps == 0)[0]
      assert all(max_classes[zero_inds] == 0)
      # max overlap > 0 => class should not be zero (must be a fg class)
      nonzero_inds = np.where(max_overlaps > 0)[0]
      #assert all(max_classes[nonzero_inds] != 0)
    else:
      #support paired datasets
      for nr, i_roidb in enumerate(roidb[i]):
        if nr == 0:
          i_roidb['image'] = imdb.image_path_at(i)
        else:
          i_roidb['image'] = imdb.image_path_at(i).replace("DAPI","mCherry")
        if not (imdb.name.startswith('coco')):
          i_roidb['width'] = sizes[i][0]
          i_roidb['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = i_roidb['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        i_roidb['max_classes'] = max_classes
        i_roidb['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        # assert all(max_classes[nonzero_inds] != 0)