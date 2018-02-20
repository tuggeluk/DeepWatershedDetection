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


def prep_im_for_blob(im, pixel_means, target_size, max_size, crop, crop_scale):
  """Mean subtract and scale an image for use in a blob."""
  # im = im.astype(np.float32, copy=False)
  # im -= pixel_means
  im_shape = im.shape
  if not crop:
    # scale if necessary
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
    # scale using pre-crop scaling factor
    #im_scale = crop_scale
    # Todo export to config file
    scale_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    im_scale = random.choice(scale_list)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    im_shape = im.shape

    # Todo export to config file
    topleft = random.uniform(0,1)<0.3

    # crop to max size if necessary
    if im_shape[0] < max_size or topleft:
      crop_0 = 0
    else:
      crop_0 = random.randint(0,im_shape[0]-max_size)

    if im_shape[1] < max_size or topleft:
      crop_1 = 0
    else:
      crop_1 = random.randint(0,im_shape[1]-max_size)

    crop_box = [crop_0, crop_1, crop_0+max_size, crop_1+max_size]
    im = im[crop_box[0]:crop_box[2],crop_box[1]:crop_box[3]]


  # pad to fit RefineNet #TODO fix refinenet padding problem
  y_mulity = int(np.ceil(im.shape[0] / 320.0))
  x_mulity = int(np.ceil(im.shape[1] / 320.0))
  canv = np.ones([y_mulity * 320, x_mulity * 320,3], dtype=np.uint8) * 255
  canv[0:im.shape[0], 0:im.shape[1]] = im
  im = canv

  return im, im_scale, crop_box

