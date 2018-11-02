from models.RefineNet import build_refinenet
import tensorflow as tf
from tensorflow.contrib import slim
from main.config import cfg


def build_dwd_net(input,model,num_classes,pretrained_dir,substract_mean = False):
    g, init_fn = build_refinenet(input, preset_model=model, num_classes=None, pretrained_dir=pretrained_dir,
                                 substract_mean=substract_mean)

    network_heads = dict()
    with tf.variable_scope('deep_watershed'):

        network_heads["stamp_class"] = dict()
        # class binary
        network_heads["stamp_class"]["binary"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='class_binary_'+str(x)) for x in range(0, len(g))]
        # class pred
        network_heads["stamp_class"]["softmax"] = [slim.conv2d(g[x], num_classes, [1, 1], activation_fn=None, scope='class_pred_' + str(x)) for x in range(0, len(g))]

        # direction
        network_heads["stamp_directions"] = dict()
        network_heads["stamp_directions"]["reg"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='direction_' + str(x)) for x in range(0, len(g))]

        network_heads["stamp_energy"] = dict()
        # energy marker - regression
        network_heads["stamp_energy"]["reg"] = [slim.conv2d(g[x], 1, [1, 1], activation_fn=None, scope='energy_reg_' + str(x)) for x in range(0, len(g))]
        # energy marker - logits
        network_heads["stamp_energy"]["softmax"] = [slim.conv2d(g[x], cfg.TRAIN.MAX_ENERGY, [1, 1], activation_fn=None, scope='energy_logits_' + str(x)) for x in range(0, len(g))]

        network_heads["stamp_bbox"] = dict()
        # bbox_size - reg
        network_heads["stamp_bbox"]["reg"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='bbox_reg_' + str(x)) for x in range(0, len(g))]
        # bbox_size - logits
        network_heads["stamp_bbox"]["softmax"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='bbox_logits_' + str(x)) for x in range(0, len(g))]

        return network_heads, init_fn