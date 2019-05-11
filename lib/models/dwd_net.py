from models.RefineNet import build_refinenet
import tensorflow as tf
from tensorflow.contrib import slim
#from main.config import cfg
from models.UNet import build_u_net

def build_dwd_net(input,model,num_classes,pretrained_dir, max_energy, substract_mean = False, individual_upsamp = "False", paired_mode=1,  used_heads=None, sparse_heads="False"):

    if "RefineNet" in model:
        g, init_fn = build_refinenet(input, preset_model=model, num_classes=None, pretrained_dir=pretrained_dir,
                                     substract_mean=substract_mean,individual_upsamp=individual_upsamp, paired_mode=paired_mode, used_heads=used_heads, sparse_heads=sparse_heads)
    elif "UNet" in model:
        g, variables, size_diff = build_u_net(input,tf.constant(1, dtype=tf.float32), channels=3, n_class=256,layers=5, features_root=64,individual_upsamp=individual_upsamp, paired_mode=paired_mode, used_heads=used_heads)
        init_fn = None # no pretrained models

    elif "DeepLab" in model:
        raise NotImplementedError

    else:
        print("unknown model")
        raise NotImplementedError


    if individual_upsamp != "task" and individual_upsamp != "sub_task":
        # copy along tasks
        tasks_dict = dict()
        for _,task in enumerate(used_heads):
            tasks_dict[task] = g
        g = tasks_dict

    if individual_upsamp != "sub_task":
        # copy along sub task
        pairs_list = list()
        for pair_nr in range(paired_mode):
            pairs_list.append(g)

        g = pairs_list

    network_heads_list = []
    with tf.variable_scope('deep_watershed'):

        for pair_nr in range(paired_mode):
            network_heads = dict()

            if "stamp_class" in used_heads:
                # classification
                network_heads["stamp_class"] = dict()
                # class binary
                network_heads["stamp_class"]["binary"] = [slim.conv2d(g[pair_nr]["stamp_class"][x], 2, [1, 1], activation_fn=None, scope='class_binary_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in
                                                          range(0, len(g[pair_nr]["stamp_class"]))]
                # class pred
                network_heads["stamp_class"]["softmax"] = [slim.conv2d(g[pair_nr]["stamp_class"][x], num_classes, [1, 1], activation_fn=None, scope='class_pred_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in
                                                           range(0, len(g[pair_nr]["stamp_class"]))]
            if "stamp_directions" in used_heads:
                # direction
                network_heads["stamp_directions"] = dict()
                network_heads["stamp_directions"]["reg"] = [slim.conv2d(g[pair_nr]["stamp_directions"][x], 2, [1, 1], activation_fn=None, scope='direction_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in range(0, len(g[pair_nr]["stamp_directions"]))]

            if "stamp_energy" in used_heads:
                # energy
                network_heads["stamp_energy"] = dict()
                # energy marker - regression
                network_heads["stamp_energy"]["reg"] = [slim.conv2d(g[pair_nr]["stamp_energy"][x], 1, [1, 1], activation_fn=None, scope='energy_reg_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in range(0, len(g[pair_nr]["stamp_energy"]))]
                # energy marker - logits
                network_heads["stamp_energy"]["softmax"] = [slim.conv2d(g[pair_nr]["stamp_energy"][x], max_energy, [1, 1], activation_fn=None, scope='energy_logits_' + 'pair' + str(pair_nr) + '_' + str(x)) for x
                                                            in range(0, len(g[pair_nr]["stamp_energy"]))]
            if "stamp_bbox" in used_heads:
                # bounding boxes
                network_heads["stamp_bbox"] = dict()
                # bbox_size - reg
                network_heads["stamp_bbox"]["reg"] = [slim.conv2d(g[pair_nr]["stamp_bbox"][x], 2, [1, 1], activation_fn=None, scope='bbox_reg_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in range(0, len(g[pair_nr]["stamp_bbox"]))]
                # bbox_size - logits
                network_heads["stamp_bbox"]["softmax"] = [slim.conv2d(g[pair_nr]["stamp_bbox"][x], 2, [1, 1], activation_fn=None, scope='bbox_logits_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in range(0, len(g[pair_nr]["stamp_bbox"]))]

            if "stamp_semseg" in used_heads:
                # semseg
                network_heads["stamp_semseg"] = dict()
                network_heads["stamp_semseg"]["softmax"] = [slim.conv2d(g[pair_nr]["stamp_semseg"][x], num_classes, [1, 1], activation_fn=None, scope='semseg_' + 'pair' + str(pair_nr) + '_' + str(x)) for x in
                                                            range(0, len(g[pair_nr]["stamp_semseg"]))]

            network_heads_list.append(network_heads)

    return network_heads_list, init_fn

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


