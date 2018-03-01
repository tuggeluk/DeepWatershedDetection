
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

from datasets.fcn_groundtruth import stamp_class, stamp_directions, stamp_energy, try_all_assign

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
    if args.prefetch == "True":
        data_layer = PrefetchWrapper(data_layer.forward, args.prefetch_len, args)

    # replaces keywords with function handles in training assignements
    save_objectness_function_handles(args,imdb)

    #
    # Debug stuffs
    #
    try_all_assign(data_layer,args)

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

    used_losses_and_optimizers = []
    for assign in args.training_assignements:
        losses = train_on_assignment(input,args,imdb,data_layer,saver,sess,writer,network_heads,assign,checkpoint_dir,checkpoint_name,iteration)
        used_losses_and_optimizers.append(losses)

    for comb_assign in args.combined_assignements:
        train_on_comb_assignment()

    if args.prefetch == "True":
        data_layer.kill()


def train_on_assignment(input,args,imdb,data_layer,saver,sess,writer,network_heads,assign,checkpoint_dir,checkpoint_name,iteration):
    # get groundtruth input placeholders
    gt_placeholders = get_gt_placeholders(assign,imdb)

    # define loss #TODO directional loss
    if assign["stamp_args"]["loss"] == "softmax":
        loss_components = [tf.losses.mean_squared_error(predictions=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x],
                                                        labels=gt_placeholders[x]) for x in range(len(assign["ds_factors"]))]
    else:
        loss_components = [safe_softmax_cross_entropy_with_logits(predictions=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x],
                                                        labels=gt_placeholders[x]) for x in range(len(assign["ds_factors"]))]


    # potentially mask out zeros
    if assign["mask_zeros"]:
        # only compute loss where GT is not zero intended for "directional donuts"
        masked_components = []
        for x in range(len(assign["ds_factors"])):
            mask = tf.squeeze(gt_placeholders[x] > 0, -1) # TODO does this squeeze make sense for directions
            masked_components.append(tf.boolean_mask(loss_components[x], mask))

        loss_components = masked_components


    # call tf.reduce mean on each loss component
    loss_components = [tf.reduce_mean(x) for x in loss_components]
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
    images_sums.append(tf.summary.image('final_predictions_' + str(i), final_pred_placeholder))
    images_summary_op = tf.summary.merge(images_sums)


    print("Start training")
    for itr in range(iteration, (iteration+args.itrs)):
        # load batch - only use batches with content
        batch_not_loaded = True
        while batch_not_loaded:
            if args.prefetch == "True":
                blob = data_layer.get_item()
            else:
                blob = data_layer.forward(args,assign)
            batch_not_loaded = len(blob["gt_boxes"].shape) != 3


        feed_dict = {input: blob["data"]}
        for i in range(len(gt_placeholders)):
            feed_dict[gt_placeholders[i]] = blob["gt_map"+str(i)]

        # train step
        _, loss_fetch = sess.run([optim,loss], feed_dict=feed_dict)

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(loss_fetch)

        if itr % args.tensorboard_interval == 0 or itr == 1:
            fetch_list = [scalar_summary_op]
            # fetch sub_predicitons
            [fetch_list.append(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][x]) for x in range(len(assign["ds_factors"]))]

            summary, fetched_maps = sess.run(fetch_list,feed_dict=feed_dict)
            writer.add_summary(summary, float(itr))

            images_feed_dict = get_images_feed_dict()
            # save images to tensorboard
            summary = sess.run([images_summary_op], feed_dict=images_feed_dict)
            writer.add_summary(summary[0], float(itr))


        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)

    iteration =  (iteration + args.itrs)
    return loss,optim


def get_images_feed_dict():
    #TODO build images for tensorflow
    raise NotImplementedError("cant build images yet")
    return None


def get_gt_placeholders(assign, imdb):
    gt_dim = assign["stamp_func"][1](-1, assign["stamp_args"], nr_classes)
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
    parser.add_argument("--crop_size", type=bytearray, default=[640,640], help="size of the image to be cropped to")
    parser.add_argument("--crop_top_left_bias", type=float, default=0.3, help="fixed probability that the crop will be from the top left corner")
    parser.add_argument("--max_edge", type=int, default=1280, help="if there is no cropping - scale such that the longest edge has this size")
    parser.add_argument("--use_flipped", type=str, default="False", help="wether or not to append Horizontally flipped images")
    parser.add_argument("--substract_mean", type=str, default="True", help="wether or not to substract the mean of the VOC images")
    parser.add_argument("--pad_to", type=int, default=320, help="pad the final image to have edge lengths that are a multiple of this - use 0 to do nothing")
    parser.add_argument("--pad_with", type=int, default=0,help="use this number to pad images")

    parser.add_argument("--prefetch", type=str, default="False", help="use additional process to fetch batches")
    parser.add_argument("--prefetch_len", type=int, default=2, help="prefetch queue len")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training") # code only works with batchsize 1!
    parser.add_argument("--continue_training", type=str, default="False", help="load checkpoint")
    parser.add_argument("--pretrain_lvl", type=str, default="no", help="What kind of pretraining to use: no,class,semseg")
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
    # energy markers
                            {'itrs': 50000,'ds_factors': [1,2,4,8], 'downsample_marker': False, 'overlap_solution': 'max',
                                 'stamp_func': 'stamp_energy', 'layer_loss_aggregate': 'min', 'mask_zeros': False,
                                 'stamp_args':{'marker_dim': None,'size_percentage': 0.8, "shape": "oval", "loss": "softmax", "energy_shape": "linear"}},
    # class markers 0.8% - size-downsample
                            {'itrs': 50000, 'ds_factors': [1, 2, 4, 8], 'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_class', 'layer_loss_aggregate': 'min', 'mask_zeros': False,
                             'stamp_args': {'marker_dim': None, 'size_percentage': 0.8, "shape": "square", "class_resolution": "class", "loss": "softmax"}},
    # direction markers 0.3 to 0.7 percent, downsample
                            {'itrs': 50000, 'ds_factors': [1, 2, 4, 8], 'downsample_marker': True, 'overlap_solution': 'nearest',
                             'stamp_func': 'stamp_directions', 'layer_loss_aggregate': 'min', 'mask_zeros': False,
                             'stamp_args': {'marker_dim': None, 'size_percentage': 0.7, 'hole': 0.3, 'loss': "reg"}}
                        ],help="configure how groundtruth is built, see datasets.fcn_groundtruth")

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