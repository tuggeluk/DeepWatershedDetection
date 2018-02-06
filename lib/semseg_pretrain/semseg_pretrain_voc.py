# pretrain fully-conv networks on semantic segmentation tasks
# training-code adapted from https://github.com/DrSleep/tensorflow-deeplab-resnet

import os
import tensorflow as tf
import numpy as np

# voc images mean
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

