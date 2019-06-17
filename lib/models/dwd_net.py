from models.RefineNet import build_refinenet
import tensorflow as tf
from tensorflow.contrib import slim
#from main.config import cfg
from models.UNet import build_u_net

def build_dwd_net(input,model,num_classes,pretrained_dir, max_energy,individual_upsamp, assigns, substract_mean = False, n_filters = 256):

    if "RefineNet" in model:
        g, init_fn = build_refinenet(input, preset_model=model, num_classes=None, pretrained_dir=pretrained_dir,
                                     substract_mean=substract_mean, individual_upsamp=individual_upsamp, n_filters=n_filters)
    # elif "UNet" in model:
    #     g, variables, size_diff = build_u_net(input,tf.constant(1, dtype=tf.float32), channels=3, n_class=256,layers=4, features_root=32,individual_upsamp=individual_upsamp, paired_mode=paired_mode, used_heads=used_heads)
    #     init_fn = None # no pretrained models

    elif "DeepLab" in model:
        raise NotImplementedError

    else:
        print("unknown model")
        raise NotImplementedError

    network_heads = dict()
    with tf.variable_scope('exit_convs'):
        for ix, us_stem in enumerate(individual_upsamp):
            for us_head in us_stem:
                assign = assigns[int(us_head.split("_")[0])]

                if "stamp_class" == assign["stamp_func"][0]:

                    if assign["stamp_args"]["loss"] == "binary":
                        # class binary
                        network_heads[us_head] = [slim.conv2d(g[ix][x], 2, [1, 1], activation_fn=None, scope='class_binary__' + us_head + '__' + str(x)) for x in
                                                                  range(0, len(g[ix]))]

                    if assign["stamp_args"]["loss"] == "reg":
                        # class pred
                        network_heads[us_head] = [slim.conv2d(g[ix][x], num_classes, [1, 1], activation_fn=None, scope='class_pred__' + us_head + '__' + str(x)) for x in
                                                                   range(0, len(g[ix]))]

                if "stamp_directions" == assign["stamp_func"][0]:
                    # direction
                    network_heads[us_head] = [slim.conv2d(g[ix][x], 2, [1, 1], activation_fn=None, scope='direction__' + us_head + '__' + str(x)) for x in range(0, len(g[ix]))]

                if "stamp_energy" == assign["stamp_func"][0]:
                    # energy
                    # energy marker - regression
                    if assign["stamp_args"]["loss"] == "reg":
                        network_heads[us_head] = [slim.conv2d(g[ix][x], 1, [1, 1], activation_fn=None, scope='energy_reg__' + us_head + '__' + str(x)) for x in range(0, len(g[ix]))]
                    if assign["stamp_args"]["loss"] == "softmax":
                        # energy marker - logits
                        network_heads[us_head] = [slim.conv2d(g[ix][x], max_energy, [1, 1], activation_fn=None, scope='energy_logits__' + us_head + '__' + str(x)) for x
                                                                    in range(0, len(g[ix]))]
                if "stamp_bbox" == assign["stamp_func"][0]:
                    # bounding boxes
                    # bbox_size - reg
                    if assign["stamp_args"]["loss"] == "reg":
                        network_heads[us_head] = [slim.conv2d(g[ix][x], 2, [1, 1], activation_fn=None, scope='bbox_reg__' + us_head + '__' + str(x)) for x in range(0, len(g[ix]))]
                    # bbox_size - logits
                    if assign["stamp_args"]["loss"] == "softmax":
                        network_heads[us_head] = [slim.conv2d(g[ix][x], 2, [1, 1], activation_fn=None, scope='bbox_logits__' + us_head + '__' + str(x)) for x in range(0, len(g[ix]))]

                if "stamp_semseg" == assign["stamp_func"][0]:
                    # semseg
                    network_heads[us_head] = [slim.conv2d(g[ix][x], num_classes, [1, 1], activation_fn=None, scope='semseg__' + us_head + '__' + str(x)) for x in
                                                                range(0, len(g[ix]))]


    return network_heads, init_fn

    # elif individual_upsamp == "task":
    #     network_heads_list = []
    #     with tf.variable_scope('deep_watershed'):
    #
    #         for pair_nr in range(paired_mode):
    #             network_heads = dict()
    #             # classification
    #             network_heads["stamp_class"] = dict()
    #             # class binary
    #             network_heads["stamp_class"]["binary"] = [slim.conv2d(g[0][x], 2, [1, 1], activation_fn=None, scope='class_binary_'+'pair'+str(pair_nr)+'_'+str(x)) for x in range(0, len(g[0]))]
    #             # class pred
    #             network_heads["stamp_class"]["softmax"] = [slim.conv2d(g[0][x], num_classes, [1, 1], activation_fn=None, scope='class_pred_' +'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[0]))]
    #
    #             # direction
    #             network_heads["stamp_directions"] = dict()
    #             network_heads["stamp_directions"]["reg"] = [slim.conv2d(g[1][x], 2, [1, 1], activation_fn=None, scope='direction_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[1]))]
    #
    #             # energy
    #             network_heads["stamp_energy"] = dict()
    #             # energy marker - regression
    #             network_heads["stamp_energy"]["reg"] = [slim.conv2d(g[2][x], 1, [1, 1], activation_fn=None, scope='energy_reg_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[2]))]
    #             # energy marker - logits
    #             network_heads["stamp_energy"]["softmax"] = [slim.conv2d(g[2][x], cfg.TRAIN.MAX_ENERGY, [1, 1], activation_fn=None, scope='energy_logits_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[2]))]
    #
    #             # bounding boxes
    #             network_heads["stamp_bbox"] = dict()
    #             # bbox_size - reg
    #             network_heads["stamp_bbox"]["reg"] = [slim.conv2d(g[3][x], 2, [1, 1], activation_fn=None, scope='bbox_reg_' +'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[3]))]
    #             # bbox_size - logits
    #             network_heads["stamp_bbox"]["softmax"] = [slim.conv2d(g[3][x], 2, [1, 1], activation_fn=None, scope='bbox_logits_' +'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[3]))]
    #
    #             # semseg
    #             network_heads["stamp_semseg"] = dict()
    #             network_heads["stamp_semseg"]["softmax"] = [slim.conv2d(g[4][x], num_classes, [1, 1], activation_fn=None, scope='semseg_' +'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g[4]))]
    #
    #             network_heads_list.append(network_heads)
    #
    #         return network_heads_list, init_fn
    #
    # else:
    #     network_heads_list = []
    #     with tf.variable_scope('deep_watershed'):
    #
    #         for pair_nr in range(paired_mode):
    #             network_heads = dict()
    #             network_heads["stamp_class"] = dict()
    #             # class binary
    #             network_heads["stamp_class"]["binary"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='class_binary_' +'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #             # class pred
    #             network_heads["stamp_class"]["softmax"] = [slim.conv2d(g[x], num_classes, [1, 1], activation_fn=None, scope='class_pred_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #
    #             # direction
    #             network_heads["stamp_directions"] = dict()
    #             network_heads["stamp_directions"]["reg"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='direction_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #
    #             network_heads["stamp_energy"] = dict()
    #             # energy marker - regression
    #             network_heads["stamp_energy"]["reg"] = [slim.conv2d(g[x], 1, [1, 1], activation_fn=None, scope='energy_reg_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #             # energy marker - logits
    #             network_heads["stamp_energy"]["softmax"] = [slim.conv2d(g[x], cfg.TRAIN.MAX_ENERGY, [1, 1], activation_fn=None, scope='energy_logits_' +'pair'+str(pair_nr)+'_'+ str(x)) for x in range(0, len(g))]
    #
    #             network_heads["stamp_bbox"] = dict()
    #             # bbox_size - reg
    #             network_heads["stamp_bbox"]["reg"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='bbox_reg_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #             # bbox_size - logits
    #             network_heads["stamp_bbox"]["softmax"] = [slim.conv2d(g[x], 2, [1, 1], activation_fn=None, scope='bbox_logits_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #
    #                         # semseg
    #             network_heads["stamp_semseg"] = dict()
    #             network_heads["stamp_semseg"]["softmax"] = [slim.conv2d(g[x], num_classes, [1, 1], activation_fn=None, scope='semseg_'+'pair'+str(pair_nr)+'_' + str(x)) for x in range(0, len(g))]
    #
    #             network_heads_list.append(network_heads)
    #
    #         return network_heads_list, init_fn


