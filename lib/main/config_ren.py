from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()


# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Whether to use all ground truth bounding boxes for training,
# For COCO, setting USE_ALL_GT to False will exclude boxes that are flagged as ''iscrowd''
__C.TRAIN.USE_ALL_GT = True

# how many softmax levels are to be used for energy
__C.TRAIN.MAX_ENERGY = 20


#
# MISC
#

# For reproducibility
__C.RNG_SEED = 3

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[104.00698793, 116.66876762, 122.67891434]]])


# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Pretrain directory
__C.PRETRAINED_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'pretrained'))

# Place outputs under an experiments directory
__C.EXP_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'experiments'))


# Place outputs under an experiments directory
__C.OUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0
