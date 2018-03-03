
import os
import tensorflow as tf
import numpy as np
import argparse
import sys
from main.config import cfg

from models.dwd_net import build_dwd_net

from datasets.factory import get_imdb
from tensorflow.contrib import slim
from main.dws_transform import perform_dws
from utils.safe_softmax_wrapper import safe_softmax_cross_entropy_with_logits
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
import utils.summary_helpers as sh
from collections import OrderedDict
from utils.prefetch_wrapper import PrefetchWrapper

from PIL import Image, ImageDraw

from datasets.fcn_groundtruth import stamp_class, stamp_directions, stamp_energy, try_all_assign,get_gt_visuals,get_map_visuals

nr_classes = None
# - make different FCN architecture available --> RefineNet, DeepLabv3, standard fcn

def main(unused_argv):
    print(args)
    iteration = 1

    np.random.seed(cfg.RNG_SEED)

    # load database
    imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val = load_database(args)

    global nr_classes
    nr_classes = len(imdb._classes)
    args.nr_classes.append(nr_classes)

    # replaces keywords with function handles in training assignements
    save_objectness_function_handles(args,imdb)

    #
    # Debug stuffs
    #
    #try_all_assign(data_layer,args,100)

    # dws_list = perform_dws(data["dws_energy"], data["class_map"], data["bbox_fcn"])
    #
    #

    # tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # input and output tensors
    if "DeepScores" in args.dataset:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        img_pred_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 1])
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"

    else:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_pred_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        resnet_dir = cfg.PRETRAINED_DIR+"/ImageNet/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/VOC2012/"


    print("Initializing Model:" + args.model)
    # model has all possible output heads (even if unused) to ensure saving and loading goes smoothly
    network_heads, init_fn = build_dwd_net(
        input, model=args.model,num_classes=nr_classes, pretrained_dir=resnet_dir, substract_mean=False)

    # init tensorflow session
    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    # load model weights
    checkpoint_dir = get_checkpoint_dir(args)
    checkpoint_name =  "backbone"
    if args.continue_training == "True":
        print("Loading checkpoint")
        saver.restore(sess, checkpoint_dir + "/" + checkpoint_name)
    else:
        if args.pretrain_lvl == "semseg":
            #load all variables except the ones in scope "deep_watershed"
            pretrained_vars = []
            for var in slim.get_model_variables():
                if "deep_watershed" not in var.name:
                    pretrained_vars.append(var)

            print("Loading network pretrained on semantic segmentation")
            loading_checkpoint_name = refinenet_dir + args.model + ".ckpt"
            init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
            init_fn(sess)
        elif args.pretrain_lvl == "class":
            print("Loading pretrained weights for level: " + args.pretrain_lvl)
            init_fn(sess)
        else:
            print("Not loading a pretrained network")

    # set up tensorboard
    writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)

    #train_on_assignment(input,args,imdb,data_layer,saver,sess,writer,network_heads,args.training_assignements[2],checkpoint_dir,checkpoint_name,iteration)

    used_losses_and_optimizers = []
    for assign in args.training_assignements:
        [losses ,optim, placeholders ,iteration] = train_on_assignment(input,args,imdb,data_layer,saver,sess,writer,network_heads,assign,checkpoint_dir,checkpoint_name,iteration)
        used_losses_and_optimizers.append([losses, optim, placeholders])

    for redo_a in args.redo_assign:
        assign_nr = redo_a["Nr"]
        redo_itr = redo_a["Itrs"]
        redo_loss = used_losses_and_optimizers[assign_nr][0]
        redo_optim = used_losses_and_optimizers[assign_nr][1]
        placeholders = used_losses_and_optimizers[assign_nr][2]
        iteration = redo_assign(input,args,imdb,data_layer,saver,sess,writer,network_heads,
                                args.training_assignements[assign_nr],checkpoint_dir,checkpoint_name,iteration,
                                redo_itr,redo_loss,redo_optim,placeholders)

    print("done :)")

    # for comb_assign in args.combined_assignements:
    #     train_on_comb_assignment()

def redo_assign(input,args,imdb,data_layer,saver,sess,writer,network_heads,assign,checkpoint_dir,checkpoint_name,iteration,redo_itr,redo_loss,redo_optim,placeholders):
    # get groundtruth input placeholders
    gt_placeholders = placeholders

    loss = redo_loss
    optim = redo_optim

    # define summary ops
    scalar_sums = []

    scalar_sums.append(tf.summary.scalar("loss: " + get_config_id(assign), loss))
    scalar_summary_op = tf.summary.merge(scalar_sums)

    images_sums = []
    images_placeholders = []

    # feature maps
    for i in range(len(assign["ds_factors"])):
        sub_prediction_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        images_placeholders.append(sub_prediction_placeholder)
        images_sums.append(tf.summary.image('sub_prediction_' + str(i), sub_prediction_placeholder))

        sub_gt_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        images_placeholders.append(sub_gt_placeholder)
        images_sums.append(tf.summary.image('sub_gt_' + str(i), sub_gt_placeholder))

    final_pred_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
    images_placeholders.append(final_pred_placeholder)
    images_sums.append(tf.summary.image('final_predictions_' + str(i), final_pred_placeholder))
    images_summary_op = tf.summary.merge(images_sums)

    if args.prefetch == "True":
        data_layer = PrefetchWrapper(data_layer.forward, args.prefetch_len, args, assign)

    print("Start training")
    for itr in range(iteration, (iteration + redo_itr)):
        print(itr)
        # load batch - only use batches with content
        batch_not_loaded = True
        while batch_not_loaded:
            blob = data_layer.forward(args, assign)
            batch_not_loaded = len(blob["gt_boxes"].shape) != 3

        feed_dict = {input: blob["data"]}
        for i in range(len(gt_placeholders)):
            feed_dict[gt_placeholders[i]] = blob["gt_map" + str(len(gt_placeholders) - i - 1)]

        # initialize variable uninitalized at this point
        sess.run(tf.global_variables_initializer())
        # train step
        _, loss_fetch = sess.run([optim, loss], feed_dict=feed_dict)

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(loss_fetch)

        if itr % args.tensorboard_interval == 0 or itr == 1:
            fetch_list = [scalar_summary_op]
            # fetch sub_predicitons
            [fetch_list.append(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x]) for x in
             range(len(assign["ds_factors"]))]

            summary = sess.run(fetch_list, feed_dict=feed_dict)
            writer.add_summary(summary[0], float(itr))

            # use predicted feature maps
            # TODO predict boxes

            gt_visuals = get_gt_visuals(blob, assign, pred_boxes=None, show=False)
            map_visuals = get_map_visuals(summary[1:], assign, show=False)
            images_feed_dict = get_images_feed_dict(assign, blob, gt_visuals, map_visuals, images_placeholders)
            # save images to tensorboard
            summary = sess.run([images_summary_op], feed_dict=images_feed_dict)
            writer.add_summary(summary[0], float(itr))

        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)

    iteration = (iteration + assign["itrs"])
    if args.prefetch == "True":
        data_layer.kill()

    return iteration

def train_on_assignment(input,args,imdb,data_layer,saver,sess,writer,network_heads,assign,checkpoint_dir,checkpoint_name,iteration):
    # get groundtruth input placeholders
    gt_placeholders = get_gt_placeholders(assign,imdb)
    debug_fetch = dict()
    # define loss #TODO directional loss
    if assign["stamp_func"][0] == "stamp_directions":
        loss_components = []
        for x in range(len(assign["ds_factors"])):
            debug_fetch[str(x)] = dict()
            # # mask, where gt is zero
            split1, split2 = tf.split(gt_placeholders[x],2,-1)
            debug_fetch[str(x)]["split1"] = split1

            mask = tf.squeeze(split1 > 0, -1)
            debug_fetch[str(x)]["mask"] = mask

            masked_pred = tf.boolean_mask(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x], mask)
            debug_fetch[str(x)]["masked_pred"]= masked_pred

            masked_gt = tf.boolean_mask(gt_placeholders[x], mask)
            debug_fetch[str(x)]["masked_gt"] = masked_gt

            # norm prediction
            norms = tf.norm(masked_pred,ord="euclidean",axis=-1,keep_dims=True)
            masked_pred = masked_pred/norms
            debug_fetch[str(x)]["masked_gt_normed"] = masked_pred

            # inner product
            inner = tf.diag_part(tf.tensordot(masked_gt,tf.transpose(masked_pred),1))
            acos_inner = tf.acos(inner)
            debug_fetch[str(x)]["acos_inner"] = acos_inner

            loss_components.append(acos_inner)
    else:
        if assign["stamp_args"]["loss"] == "softmax":
            loss_components = [tf.losses.mean_squared_error(predictions=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x],
                                                            labels=gt_placeholders[x]) for x in range(len(assign["ds_factors"]))]
        else:
            loss_components = [safe_softmax_cross_entropy_with_logits(predictions=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x],
                                                            labels=gt_placeholders[x]) for x in range(len(assign["ds_factors"]))]


    #################################################################################################################
    #     # debug directional loss
    #         # load batch - only use batches with content
    # batch_not_loaded = True
    # while batch_not_loaded:
    #     if args.prefetch == "True":
    #         blob = data_layer.get_item()
    #     else:
    #         blob = data_layer.forward(args, assign)
    #     batch_not_loaded = len(blob["gt_boxes"].shape) != 3
    #
    #     feed_dict = {input: blob["data"]}
    #     for i in range(len(gt_placeholders)):
    #         feed_dict[gt_placeholders[i]] = blob["gt_map" + str(len(gt_placeholders)-i-1)]
    #
    # 1==1
    # # train step
    #
    # [split] = sess.run([debug_fetch[str(x)]["split1"]], feed_dict=feed_dict)
    # [pred] = sess.run([network_heads[assign["stamp_func"][0]][x]], feed_dict=feed_dict)
    # [mask] = sess.run([debug_fetch[str(x)]["mask"]], feed_dict=feed_dict)
    # [masked_gt_normed] = sess.run([debug_fetch[str(x)]["masked_gt_normed"]], feed_dict=feed_dict)
    # [acos_inner] = sess.run([debug_fetch[str(x)]["acos_inner"]], feed_dict=feed_dict)
    #
    #
    # debug_fetch[str(x)].keys()
    #     # for i in range(mask_gt_fetch.shape[0]):
    #     #     print(np.inner(mask_gt_fetch[i],mask_pred_fetch[i]))
    #################################################################################################################

        # potentially mask out zeros
        if assign["mask_zeros"]:
            # only compute loss where GT is not zero intended for "directional donuts"
            masked_components = []
            for x in range(len(assign["ds_factors"])):
                mask = tf.squeeze(gt_placeholders[x] > 0, -1)
                masked_components.append(tf.boolean_mask(loss_components[x], mask))

            loss_components = masked_components


    # call tf.reduce mean on each loss component
    loss_components = [tf.reduce_mean(x) for x in loss_components]

    # replace with loss of last layer if is nan (can happen if last mask has no directions on it)
    loss_components = [tf.cond(tf.is_nan(x), lambda: loss_components[len(loss_components)-1], lambda: x) for x in loss_components]

    stacked_components = tf.stack(loss_components)



    if assign["layer_loss_aggregate"] == "min":
        loss = tf.reduce_min(stacked_components)
    elif assign["layer_loss_aggregate"] == "avg":
        loss = tf.reduce_mean(stacked_components)
    else:
        raise NotImplementedError("unknown layer aggregate")



    # init optimizer
    var_list = [var for var in tf.trainable_variables()]
    optim = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=var_list)

    # define summary ops
    scalar_sums = []

    for i in range(len(assign["ds_factors"])):
        scalar_sums.append(tf.summary.scalar("sub_loss_"+str(i)+": " + get_config_id(assign), loss_components[i]))

    scalar_sums.append(tf.summary.scalar("loss: "+get_config_id(assign), loss))
    scalar_summary_op = tf.summary.merge(scalar_sums)

    images_sums = []
    images_placeholders = []

    # feature maps
    for i in range(len(assign["ds_factors"])):
        sub_prediction_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        images_placeholders.append(sub_prediction_placeholder)
        images_sums.append(tf.summary.image('sub_prediction_'+str(i), sub_prediction_placeholder))

        sub_gt_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
        images_placeholders.append(sub_gt_placeholder)
        images_sums.append(tf.summary.image('sub_gt_' + str(i), sub_gt_placeholder))

    final_pred_placeholder = tf.placeholder(tf.uint8, shape=[1, None, None, 3])
    images_placeholders.append(final_pred_placeholder)
    images_sums.append(tf.summary.image('final_predictions_' + str(i), final_pred_placeholder))
    images_summary_op = tf.summary.merge(images_sums)

    if args.prefetch == "True":
        data_layer = PrefetchWrapper(data_layer.forward, args.prefetch_len, args, assign)

    sess.run(tf.global_variables_initializer())
    print("Start training")
    for itr in range(iteration, (iteration+assign["itrs"])):
        print(itr)
        # load batch - only use batches with content
        batch_not_loaded = True
        while batch_not_loaded:
            blob = data_layer.forward(args,assign)
            batch_not_loaded = len(blob["gt_boxes"].shape) != 3


        feed_dict = {input: blob["data"]}
        for i in range(len(gt_placeholders)):
            feed_dict[gt_placeholders[i]] = blob["gt_map" + str(len(gt_placeholders) - i - 1)]

        # initialize variable uninitalized at this point
        #sess.run(tf.global_variables_initializer())
        # train step
        _, loss_fetch = sess.run([optim,loss], feed_dict=feed_dict)

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(loss_fetch)

        if itr % args.tensorboard_interval == 0 or itr == 1:
            fetch_list = [scalar_summary_op]
            # fetch sub_predicitons
            [fetch_list.append(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x]) for x in range(len(assign["ds_factors"]))]

            summary = sess.run(fetch_list,feed_dict=feed_dict)
            writer.add_summary(summary[0], float(itr))

            # use predicted feature maps
            # TODO predict boxes

            gt_visuals = get_gt_visuals(blob, assign, pred_boxes=None, show=False)
            map_visuals = get_map_visuals(summary[1:],assign,show=False)
            images_feed_dict = get_images_feed_dict(assign,blob,gt_visuals,map_visuals,images_placeholders)
            # save images to tensorboard
            summary = sess.run([images_summary_op], feed_dict=images_feed_dict)
            writer.add_summary(summary[0], float(itr))


        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)

    iteration = (iteration + assign["itrs"])
    if args.prefetch == "True":
        data_layer.kill()

    return loss,optim,gt_placeholders, iteration


def get_images_feed_dict(assign,blob,gt_visuals,map_visuals,images_placeholders):
    feed_dict = dict()
    for i in range(len(assign["ds_factors"])*2):
        if i%2 ==0:
            # prediction
            feed_dict[images_placeholders[i]] = map_visuals[i/2]

        else:
            feed_dict[images_placeholders[i]] = gt_visuals[i/2]


    for key in feed_dict.keys():
        feed_dict[key] = np.expand_dims(feed_dict[key], 0)

    feed_dict[images_placeholders[len(images_placeholders)-1]] = blob["data"].astype(np.uint8)



    return feed_dict


def get_gt_placeholders(assign, imdb):
    gt_dim = assign["stamp_func"][1](None, assign["stamp_args"], nr_classes)
    return [tf.placeholder(tf.float32, shape=[None, None, None, gt_dim]) for x in assign["ds_factors"]]


def train_on_comb_assignment():
    return


def get_config_id(assign):
    return assign["stamp_func"][0]+"_"+ assign["stamp_args"]["loss"]



def get_checkpoint_dir(args):
    # assemble path
    if "DeepScores" in args.dataset:
        image_mode = "music"
    else:
        image_mode = "realistic"
    tbdir = cfg.EXP_DIR + "/" + image_mode +"/"+"pretrain_lvl_"+args.pretrain_lvl+"/" + args.model
    if not os.path.exists(tbdir):
        os.makedirs(tbdir)
    runs_dir = os.listdir(tbdir)
    if args.continue_training == "True":
        tbdir = tbdir + "/" + "run_" + str(len(runs_dir)-1)
    else:
        tbdir = tbdir+"/"+"run_"+str(len(runs_dir))
        os.makedirs(tbdir)
    return tbdir


def get_training_roidb(imdb, use_flipped):
  """Returns a roidb (Region of Interest database) for use in training."""
  if use_flipped:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb


def save_objectness_function_handles(args, imdb):
    FUNCTION_MAP = {'stamp_directions':stamp_directions,
                    'stamp_energy': stamp_energy,
                    'stamp_class': stamp_class}

    for obj_setting in args.training_assignements:
        obj_setting["stamp_func"] = [obj_setting["stamp_func"], FUNCTION_MAP[obj_setting["stamp_func"]]]

    return args

def load_database(args):
    print("Setting up image database: " + args.dataset)
    imdb = get_imdb(args.dataset)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb, args.use_flipped == "True")
    print('{:d} roidb entries'.format(len(roidb)))

    if args.dataset_validation != "no":
        print("Setting up validation image database: " + args.dataset_validation)
        imdb_val = get_imdb(args.dataset_validation)
        print('Loaded dataset `{:s}` for validation'.format(imdb_val.name))
        roidb_val = get_training_roidb(imdb_val, False)
        print('{:d} roidb entries'.format(len(roidb_val)))
    else:
        imdb_val = None
        roidb_val = None


    data_layer = RoIDataLayer(roidb, imdb.num_classes)

    if roidb_val is not None:
        data_layer_val = RoIDataLayer(roidb_val, imdb_val.num_classes, random=True)

    return imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val

def get_nr_classes():
    return nr_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--scale_list", type=list, default=[0.9,1,1.1], help="global scaling factor randomly chosen from this list")
    parser.add_argument("--crop", type=str, default="False", help="should images be cropped")
    parser.add_argument("--crop_size", type=bytearray, default=[160,160], help="size of the image to be cropped to")
    parser.add_argument("--crop_top_left_bias", type=float, default=0.3, help="fixed probability that the crop will be from the top left corner")
    parser.add_argument("--max_edge", type=int, default=1280, help="if there is no cropping - scale such that the longest edge has this size")
    parser.add_argument("--use_flipped", type=str, default="False", help="wether or not to append Horizontally flipped images")
    parser.add_argument("--substract_mean", type=str, default="True", help="wether or not to substract the mean of the VOC images")
    parser.add_argument("--pad_to", type=int, default=160, help="pad the final image to have edge lengths that are a multiple of this - use 0 to do nothing")
    parser.add_argument("--pad_with", type=int, default=0,help="use this number to pad images")

    parser.add_argument("--prefetch", type=str, default="True", help="use additional process to fetch batches")
    parser.add_argument("--prefetch_len", type=int, default=2, help="prefetch queue len")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training") # code only works with batchsize 1!
    parser.add_argument("--continue_training", type=str, default="False", help="load checkpoint")
    parser.add_argument("--pretrain_lvl", type=str, default="class", help="What kind of pretraining to use: no,class,semseg")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
    parser.add_argument("--dataset", type=str, default="voc_2012_train", help="DeepScores, voc or coco")
    parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")
    parser.add_argument("--print_interval", type=int, default=100, help="after how many iterations is tensorboard updated")
    parser.add_argument("--tensorboard_interval", type=int, default=100, help="after how many iterations is tensorboard updated")
    parser.add_argument("--save_interval", type=int, default=2000, help="after how many iterations are the weights saved")
    parser.add_argument("--nr_classes", type=list, default=[],help="ignore, will be overwritten by program")

    parser.add_argument('--model', type=str, default="RefineNet-Res101", help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152")

    #parser.add_argument('--training_regime', type=OrderedDict, default={'pre_energy1': '2000', 'energy': '1000', 'tot': '4'}, help="Training regime: how many iterations are to be trained on which loss")
    parser.add_argument('--itrs', type=int, default=50000, help="nr of iterations")
    parser.add_argument('--segment_style', type=str, default="marker", help="network has to detect a marker or the whole bbox")
    parser.add_argument('--segment_resolution', type=str, default="class", help="binary,class or regression (for Centerness-energy)")


    parser.add_argument('--training_assignements', type=list,
                        default=[
    # direction markers 0.3 to 0.7 percent, downsample
                            {'itrs': 10000, 'ds_factors': [1,8,16,32], 'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_directions', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
                             'stamp_args': {'marker_dim': None, 'size_percentage': 0.7,"shape": "oval", 'hole': None, 'loss': "reg"}},
    # energy markers
                            {'itrs': 10000,'ds_factors': [1,8,16,32], 'downsample_marker': False, 'overlap_solution': 'max',
                                 'stamp_func': 'stamp_energy', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
                                 'stamp_args':{'marker_dim': (12,9),'size_percentage': 0.8, "shape": "oval", "loss": "softmax", "energy_shape": "linear"}},
    # class markers 0.8% - size-downsample
                            {'itrs': 10000, 'ds_factors': [1,8,16,32], 'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_class', 'layer_loss_aggregate': 'avg', 'mask_zeros': False,
                             'stamp_args': {'marker_dim': None, 'size_percentage': 0.8, "shape": "square", "class_resolution": "class", "loss": "softmax"}}

                        ],help="configure how groundtruth is built, see datasets.fcn_groundtruth")


    parser.add_argument('--redo_assign', type=list,
                        default=[
                            {"Nr": 0, "Itrs": 10000},
                            {"Nr": 1, "Itrs": 10000},
                            {"Nr": 2, "Itrs": 10000},

                            {"Nr": 0, "Itrs": 10000},
                            {"Nr": 1, "Itrs": 10000},
                            {"Nr": 2, "Itrs": 10000}

                        ], help="configure how assignements get repeated")

    parser.add_argument('--combined_assignements', type=list,
                        default=[],help="configure how groundtruth is built, see datasets.fcn_groundtruth")

    args, unparsed = parser.parse_known_args()


    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



 #
 #
 # with tf.variable_scope('deep_watershed'):
 #        print("Using loss:")
 #        # marker Loss
 #        if args.segment_resolution == "regression":
 #            l_mark_0 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=markers[0], labels=marker_0))
 #            l_mark_1 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=markers[1], labels=marker_1))
 #            l_mark_2 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=markers[2], labels=marker_2))
 #            l_mark_3 = tf.reduce_mean(tf.losses.mean_squared_error(predictions=markers[3], labels=marker_3))
 #        elif args.segment_resolution == "binary" or args.segment_resolution == "class":
 #            l_mark_0 = tf.reduce_mean(safe_softmax_cross_entropy_with_logits(logits=markers[0], labels=marker_0))
 #            l_mark_1 = tf.reduce_mean(safe_softmax_cross_entropy_with_logits(logits=markers[1], labels=marker_1))
 #            l_mark_2 = tf.reduce_mean(safe_softmax_cross_entropy_with_logits(logits=markers[2], labels=marker_2))
 #            l_mark_3 = tf.reduce_mean(safe_softmax_cross_entropy_with_logits(logits=markers[3], labels=marker_3))
 #
 #
 #        stacked_markers = tf.stack([l_mark_0,l_mark_1,l_mark_2,l_mark_3])
 #        mean_stacked_loss = tf.reduce_mean(stacked_markers)
 #        min_stacked_loss = tf.reduce_min(stacked_markers)
 #
 #        energy_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=dws_energy, labels=label_dws_energy))
 #
 #        # energy_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=dws_energy, labels=label_dws_energy))
 #        #
 #        # dws_mask = tf.squeeze(label_dws_energy >= 0, -1)
 #        #
 #        # class_masked_logits = tf.boolean_mask(class_logits, dws_mask)
 #        # class_masked_labels = tf.boolean_mask(label_class, dws_mask)
 #        # class_loss = tf.reduce_mean(safe_softmax_cross_entropy_with_logits(logits=class_masked_logits, labels=class_masked_labels))
 #        #
 #        # bbox_masked_predictions = tf.boolean_mask(bbox_size, dws_mask)
 #        # class_masked_labels = tf.boolean_mask(label_bbox, dws_mask)
 #        # box_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=bbox_masked_predictions, labels=class_masked_labels))
 #        #
 #        # ec_loss = tf.add(energy_loss * 1.0, class_loss * 0.5)
 #        # tot_loss = tf.add(ec_loss * 1.0, box_loss * 0.5)
 #
 #
 #        print("Init optimizers")
 #        var_list = [var for var in tf.trainable_variables()]
 #        opt_energy = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(energy_loss, var_list=var_list)
 #        opt_min = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(min_stacked_loss, var_list=var_list)
 #        opt_mean = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(mean_stacked_loss,var_list=var_list)
 #
 #        #opt_ec = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(ec_loss, var_list=var_list)
 #        #opt_tot = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(tot_loss, var_list=var_list)
 #
 #        print("Init Summary")
 #        scalar_sums = []
 #        scalar_sums.append(tf.summary.scalar("energy_loss", energy_loss))
 #        scalar_sums.append(tf.summary.scalar("mean_stacked", mean_stacked_loss))
 #        scalar_sums.append(tf.summary.scalar("min_stacked", min_stacked_loss))
 #        scalar_summary_op = tf.summary.merge(scalar_sums)
 #
 #        images_sums = []
 #        images_sums.append(tf.summary.image('Energy_Map', img_energy_placeholder))
 #        images_sums.append(tf.summary.image('Marker_Maps', img_marker_placeholder))
 #        images_sums.append(tf.summary.image('Prediction', img_pred_placeholder))
 #        images_summary_op = tf.summary.merge(images_sums)
 #
 #




    # leave this for later
    # print("add-prediciton to tensorboard")
    # # compute prediction
    # pred_class = np.argmax(pred_class_logits, axis=3)
    # dws_list = perform_dws(pred_energy, pred_class, pred_bbox)
    #
    # # build images
    # # rescale
    # # pred_scaled = pred_foreground[0] + np.abs(np.min(pred_foreground[0]))
    # # pred_scaled = pred_scaled / np.max(pred_scaled)*255
    # # np.argmax(, axis=None, out=None)
    # pred_scaled = np.argmax(pred_foreground[0], axis=-1, out=None)
    # pred_scaled = np.expand_dims(pred_scaled, -1) * 255
    #
    # orig_scaled = np.argmax(blob["foreground"][0], axis=-1, out=None)
    # orig_scaled = np.expand_dims(orig_scaled, -1) * 255
    #
    # conc_array = np.concatenate((pred_scaled, orig_scaled), 0)
    # energy_array = np.squeeze(conc_array.astype("uint8"))
    # energy_array = np.expand_dims(np.expand_dims(energy_array, -1), 0)
    #
    # # switch bgr to rgb
    # im_rgb = blob["data"][0][:, :, [2, 1, 0]] + cfg.PIXEL_MEANS[:, :, [2, 1, 0]]
    # im = Image.fromarray(im_rgb.astype("uint8"))
    # draw = ImageDraw.Draw(im)
    # # overlay GT boxes
    # for row in blob["gt_boxes"][0]:
    #     draw.rectangle(((row[0], row[1]), (row[2], row[3])), outline="green")
    # for row in dws_list:
    #     draw.rectangle(((row[0], row[1]), (row[2], row[3])), outline="red")
    # im_array = np.array(im).astype("uint8")
    # im_array = np.expand_dims(im_array, 0)
    #
    # if len(im_array.shape) < 4:
    #     im_array = np.expand_dims(im_array, -1)


    # if "DeepScores" in args.dataset:
    #     blob["data"] = np.expand_dims(np.mean(blob["data"], -1), -1)
    #     # one-hot class labels
    #     blob["class_map"] = np.eye(imdb.num_classes)[blob["class_map"][:, :, :, -1]]
    #
    # if "voc" in args.dataset:
    #     # one-hot class labels
    #     blob["class_map"] = np.eye(imdb.num_classes)[blob["class_map"][:, :, :, -1]]
    #     blob["foreground"] = np.eye(2)[blob["foreground"][:, :, :, -1]]