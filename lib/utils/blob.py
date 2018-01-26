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


def im_list_to_blob(ims):
  """Convert a list of images into a network input.

  Assumes images are already prepared (means subtracted, BGR order, ...).
  """
  max_shape = np.array([im.shape for im in ims]).max(axis=0)
  num_images = len(ims)
  blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                  dtype=np.float32)
  for i in range(num_images):
    im = ims[i]
    blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

  return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size, crop):
  """Mean subtract and scale an image for use in a blob."""
  im = im.astype(np.float32, copy=False)
  im -= pixel_means
  im_shape = im.shape
  if not crop:
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    crop_box = [0,0,im_shape[0],im_shape[1]]

  else:
    im_scale = 1
    if im_shape[0] < max_size:
      crop_0 = 0
    else:
      crop_0 = random.randint(0,im_shape[0]-max_size)

    if im_shape[1] < max_size:
      crop_1 = 0
    else:
      crop_1 = random.randint(0,im_shape[1]-max_size)

    crop_box = [crop_0, crop_1, crop_0+max_size, crop_1+max_size]
    im = im[crop_box[0]:crop_box[2],crop_box[1]:crop_box[3]]



  return im, im_scale,crop_box
