from models.RefineNet import build_refinenet
import tensorflow as tf
from tensorflow.contrib import slim


def build_dwd_net(input,model,num_classes,pretrained_dir,substract_mean = False):
    g, init_fn = build_refinenet(input, preset_model=model, num_classes=None, pretrained_dir=pretrained_dir,
                                 substract_mean=substract_mean)

    with tf.variable_scope('deep_watershed'):
        dws_energy = slim.conv2d(g[3], 1, [1, 1], activation_fn=None, scope='dws_energy')
        class_logits = slim.conv2d(g[3], num_classes, [1, 1], activation_fn=None, scope='logits')
        bbox_size = slim.conv2d(g[3], 2, [1, 1], activation_fn=None, scope='dws_size')


        return [dws_energy,class_logits,bbox_size], init_fn