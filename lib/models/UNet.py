# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret, taken from: https://github.com/jakeret/tf_unet/
modified by: tugg
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf


from models.unet_utils import (weight_variable, weight_variable_devonc, bias_variable,
                            conv2d, deconv2d, max_pool, crop_and_concat)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def build_u_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, individual_upsamp="False", paired_mode=1,
                used_heads=None):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    in_node = x
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000
    size = in_size

    # down layers
    for layer in range(0, layers):
        with tf.name_scope("unet_down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            conv1 = conv2d(in_node, w1, b1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
            dw_h_convs[layer] = tf.nn.relu(conv2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 2 * 2 * (filter_size // 2) # valid conv
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= pool_size



    if used_heads is None:
        us_stages = ["energy", "direction", "classes", "bbox", "semseg"]
    else:
        us_stages = used_heads

    if individual_upsamp == "sub_task":
        g_outer_list = list()
        for sub_b in range(paired_mode):
            g_list = dict()
            for stage in us_stages:
                in_node = dw_h_convs[layers - 1]
                g = []
                for layer in range(layers - 2, -1, -1):
                    with tf.name_scope("unet_up_conv_{}_stage_{}_pair_{}".format(str(layer), stage, str(sub_b))):
                        features = 2 ** (layer + 1) * features_root
                        stddev = np.sqrt(2 / (filter_size ** 2 * features))

                        wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
                        bd = bias_variable([features // 2], name="bd")
                        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                        deconv[layer] = h_deconv_concat

                        w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                        w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                        b1 = bias_variable([features // 2], name="b1")
                        b2 = bias_variable([features // 2], name="b2")

                        conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
                        h_conv = tf.nn.relu(conv1)
                        conv2 = conv2d(h_conv, w2, b2, keep_prob)
                        in_node = tf.nn.relu(conv2)
                        up_h_convs[layer] = in_node

                        g.append(in_node)

                        weights.append((w1, w2))
                        biases.append((b1, b2))
                        convs.append((conv1, conv2))

                        size *= pool_size
                        size -= 2 * 2 * (filter_size // 2)  # valid conv
                g_list[stage] = g
            g_outer_list.append(g_list)
        output_map = g_outer_list
    elif individual_upsamp == "task":
        g_list = dict()
        for stage in us_stages:
            in_node = dw_h_convs[layers - 1]
            g = []
            for layer in range(layers - 2, -1, -1):
                with tf.name_scope("unet_up_conv_{}_stage_{}_pair_{}".format(str(layer), stage, "0")):
                    features = 2 ** (layer + 1) * features_root
                    stddev = np.sqrt(2 / (filter_size ** 2 * features))

                    wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
                    bd = bias_variable([features // 2], name="bd")
                    h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                    h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                    deconv[layer] = h_deconv_concat

                    w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                    w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                    b1 = bias_variable([features // 2], name="b1")
                    b2 = bias_variable([features // 2], name="b2")

                    conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
                    h_conv = tf.nn.relu(conv1)
                    conv2 = conv2d(h_conv, w2, b2, keep_prob)
                    in_node = tf.nn.relu(conv2)
                    up_h_convs[layer] = in_node

                    g.append(in_node)

                    weights.append((w1, w2))
                    biases.append((b1, b2))
                    convs.append((conv1, conv2))

                    size *= pool_size
                    size -= 2 * 2 * (filter_size // 2)  # valid conv
            g_list[stage] = g
        output_map = g
    else:
        # up layers
        in_node = dw_h_convs[layers - 1]
        g = []
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("unet_up_conv_{}_stage_{}_pair_{}".format(str(layer), "df", "d")):
                features = 2 ** (layer + 1) * features_root
                stddev = np.sqrt(2 / (filter_size ** 2 * features))

                wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
                bd = bias_variable([features // 2], name="bd")
                h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
                h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
                deconv[layer] = h_deconv_concat

                w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
                w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
                b1 = bias_variable([features // 2], name="b1")
                b2 = bias_variable([features // 2], name="b2")

                conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
                h_conv = tf.nn.relu(conv1)
                conv2 = conv2d(h_conv, w2, b2, keep_prob)
                in_node = tf.nn.relu(conv2)
                up_h_convs[layer] = in_node

                g.append(in_node)

                weights.append((w1, w2))
                biases.append((b1, b2))
                convs.append((conv1, conv2))

                size *= pool_size
                size -= 2 * 2 * (filter_size // 2)  # valid conv
        output_map = g




    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)