from models.RefineNet import build_refinenet
import tensorflow as tf
from tensorflow.contrib import slim


def build_dwd_net(args,input,model,num_classes,pretrained_dir,substract_mean = False):
    g, init_fn = build_refinenet(input, preset_model=model, num_classes=None, pretrained_dir=pretrained_dir,
                                 substract_mean=substract_mean)

    with tf.variable_scope('deep_watershed'):
        if args.segment_resolution == "regression":
            markers = [slim.conv2d(g[0], 1, [1, 1], activation_fn=None, scope='dws_mark_0'),
                       slim.conv2d(g[1], 1, [1, 1], activation_fn=None, scope='dws_mark_1'),
                       slim.conv2d(g[2], 1, [1, 1], activation_fn=None, scope='dws_mark_2'),
                       slim.conv2d(g[3], 1, [1, 1], activation_fn=None, scope='dws_mark_3')]
        elif args.segment_resolution == "binary":
            markers = [slim.conv2d(g[0], 2, [1, 1], activation_fn=None, scope='dws_mark_0'),
                       slim.conv2d(g[1], 2, [1, 1], activation_fn=None, scope='dws_mark_1'),
                       slim.conv2d(g[2], 2, [1, 1], activation_fn=None, scope='dws_mark_2'),
                       slim.conv2d(g[3], 2, [1, 1], activation_fn=None, scope='dws_mark_3')]
        elif args.segment_resolution == "class":
            markers = [slim.conv2d(g[0], num_classes, [1, 1], activation_fn=None, scope='dws_mark_0'),
                       slim.conv2d(g[1], num_classes, [1, 1], activation_fn=None, scope='dws_mark_1'),
                       slim.conv2d(g[2], num_classes, [1, 1], activation_fn=None, scope='dws_mark_2'),
                       slim.conv2d(g[3], num_classes, [1, 1], activation_fn=None, scope='dws_mark_3')]

        dws_energy = slim.conv2d(g[3], 1, [1, 1], activation_fn=None, scope='dws_energy')
        class_logits = slim.conv2d(g[3], num_classes, [1, 1], activation_fn=None, scope='logits')
        bbox_size = slim.conv2d(g[3], 2, [1, 1], activation_fn=None, scope='dws_size')
        foreground = slim.conv2d(g[3], 2, [1, 1], activation_fn=None, scope='foreground')


        return [markers,foreground,dws_energy,class_logits,bbox_size], init_fn