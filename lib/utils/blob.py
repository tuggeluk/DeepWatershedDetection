# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick - extended by Lukas Tuggener
# --------------------------------------------------------

"""Blob helper functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random
from PIL import Image


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  if len(ims[0].shape) == 2 :
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                    dtype=np.float32)
  else:
    blob = np.zeros((num_images, max_shape[0], max_shape[1], ims[0].shape[2]),
                    dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob


def compute_scalings(global_scale, roidb, args):
  im = Image.open(roidb['image'])
  im = np.array(im, dtype=np.float32)

  # do global scaling
  im = cv2.resize(im, None, None, fx=global_scale, fy=global_scale,
                  interpolation=cv2.INTER_LINEAR)

  im_size_max = np.max(im.shape[0:2])
  # Prevent the biggest axis from being more than MAX_SIZE
  if im_size_max > args.max_edge:
    if not args.crop == "True":
      # scale down if bigger than max size
      re_scale = (float(args.max_edge) / float(im_size_max))
      im = cv2.resize(im, None, None, fx=re_scale, fy=re_scale,
                      interpolation=cv2.INTER_LINEAR)
      global_scale = global_scale * re_scale
      crop_box = [0, 0, im.shape[0], im.shape[1]]
    else:
      # Crop image
      topleft = random.uniform(0, 1) < args.crop_top_left_bias

      # crop to max size if necessary
      if im.shape[0] <= args.max_edge or topleft:
        crop_0 = 0
      else:
        crop_0 = random.randint(0, im.shape[0] - args.max_edge)

      if im.shape[1] <= args.max_edge or topleft:
        crop_1 = 0
      else:
        crop_1 = random.randint(0, im.shape[1] - args.max_edge)

      crop_box = [crop_0, crop_1, min(crop_0 + args.max_edge, im.shape[0]), min(crop_1 + args.max_edge, im.shape[1])]
  else:
    crop_box = [0, 0, im.shape[0], im.shape[1]]

  return [global_scale, crop_box]
