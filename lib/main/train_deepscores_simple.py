
import os
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,os.path.dirname(__file__)[:-4])
from main.config import cfg

from models.dwd_net import build_dwd_net

from datasets.factory import get_imdb
from tensorflow.contrib import slim
from utils.safe_softmax_wrapper import safe_softmax_cross_entropy_with_logits
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.prefetch_wrapper import PrefetchWrapper
import argparse
from PIL import Image

from datasets.fcn_groundtruth import stamp_class, stamp_directions, stamp_energy, stamp_bbox,\
    try_all_assign,get_gt_visuals,get_map_visuals

from models.RefineNet import build_refinenet
# - make different FCN architecture available --> RefineNet, DeepLabv3, standard fcn

def main(parsed):

    np.random.seed(cfg.RNG_SEED)

    # load database
    imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val = load_database(args)

    global nr_classes
    nr_classes = len(imdb._classes)
    args.nr_classes.append(nr_classes)

    # replaces keywords with function handles in training assignements
    # save_objectness_function_handles(args, imdb)



    # tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # input and output tensors
    if "DeepScores" in args.dataset:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"

    num_classes = len(imdb._classes)

    print("Initializing Model:" + args.model)
    # model has all possible output heads (even if unused) to ensure saving and loading goes smoothly

    network, init_fn = build_refinenet(input, preset_model = args.model, num_classes= len(imdb._classes),pretrained_dir=resnet_dir, substract_mean=False)

    output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))

    opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in
                                                                                               tf.trainable_variables()])


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
                if not("deep_watershed" in var.name or "gt_feed_head" in var.name):
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

    assign = {'ds_factors': [1], 'downsample_marker': True, 'overlap_solution': 'no',
              'stamp_func': 'stamp_class', 'layer_loss_aggregate': 'avg', 'mask_zeros': True,
              'stamp_args': {'marker_dim': (9,9), 'size_percentage': 1, "shape": "square", "class_resolution": "class",
                             "loss": "softmax"}}

    assign["stamp_func"] = [assign["stamp_func"],stamp_class]

    for itr in range(1, (5000)):


        batch_not_loaded = True
        while batch_not_loaded:
            blob = data_layer.forward(args, assign, None)
            batch_not_loaded = len(blob["gt_boxes"].shape) != 3

        train_images = blob["data"]
        train_annotations = blob["gt_map0"]


        # im_data = np.concatenate([train_images[0], train_images[0], np.expand_dims(train_annotations[0, :, :, 0], -1)*255], -1)
        # im = Image.fromarray(im_data.astype(np.uint8))
        # im.save(sys.argv[0][:-31]+"overlayed_gt.png")
        #
        # dat_argmax = np.argmax(train_annotations[0], -1)
        # dat_argmax[dat_argmax == 0] = 255
        # im_argmax = Image.fromarray(dat_argmax.astype(np.uint8))
        # im_argmax.save(sys.argv[0][:-31] + "argmax_gt.png")

        _, current = sess.run([opt, loss], feed_dict={input: train_images, output: train_annotations})

        if itr == 1:
            print("initial loss" + str(current))

        if itr % 21 == 0:
            print("loss of current batch:" + str(current))

        if itr % 2001 == 0:
            print("saving weights")



    # execute tasks

    print("done :)")



def get_config_id(assign):
    return assign["stamp_func"][0]+"_"+ assign["stamp_args"]["loss"]



def get_checkpoint_dir(args):
    # assemble path

    image_mode = "music"
    tbdir = cfg.EXP_DIR + "/"+"simple_" + image_mode +"/"+"pretrain_lvl_"+args.pretrain_lvl+"/" + args.model
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
                    'stamp_class': stamp_class,
                    'stamp_bbox': stamp_bbox
                    }

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


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--scale_list", type=list, default=[0.5],
                      help="global scaling factor randomly chosen from this list")
  parser.add_argument("--crop", type=str, default="True", help="should images be cropped")
  parser.add_argument("--crop_top_left_bias", type=float, default=0.3,
                      help="fixed probability that the crop will be from the top left corner")
  parser.add_argument("--max_edge", type=int, default=800,
                      help="if there is no cropping - scale such that the longest edge has this size / if there is cropping crop to max_edge * max_edge")
  parser.add_argument("--use_flipped", type=str, default="False",
                      help="wether or not to append Horizontally flipped images")
  parser.add_argument("--substract_mean", type=str, default="False",
                      help="wether or not to substract the mean of the VOC images")
  parser.add_argument("--pad_to", type=int, default=160,
                      help="pad the final image to have edge lengths that are a multiple of this - use 0 to do nothing")
  parser.add_argument("--pad_with", type=int, default=0, help="use this number to pad images")

  parser.add_argument("--prefetch", type=str, default="True", help="use additional process to fetch batches")
  parser.add_argument("--prefetch_len", type=int, default=7, help="prefetch queue len")

  parser.add_argument("--batch_size", type=int, default=1,
                      help="batch size for training")

  parser.add_argument("--dataset", type=str, default="DeepScores_2017_train", help="DeepScores, voc or coco")
  parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug",
                      help="DeepScores, voc, coco or no - validation set")

  parser.add_argument("--nr_classes", type=list, default=[], help="ignore, will be overwritten by program")

  parser.add_argument('--model', type=str, default="RefineNet-Res101",
                      help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152")
  parser.add_argument("--pretrain_lvl", type=str, default="semseg",
                      help="What kind of pretraining to use: no,class,semseg")
  parser.add_argument("--continue_training", type=str, default="False", help="load checkpoint")
  args, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


