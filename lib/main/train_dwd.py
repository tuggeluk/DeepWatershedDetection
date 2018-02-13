
import os
import tensorflow as tf
import numpy as np
import argparse
import sys
from main.config import cfg
from models.RefineNet import build_refinenet
from datasets.factory import get_imdb

from tensorflow.contrib import slim

import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer

# Training regime
# - make different FCN architecture available --> RefineNet, DeepLabv3, standard fcn
# - pretrain on classification i.e. make classification loss available
#
# - try high dimensional loss
# - try regression loss
# - with and without pretraining on semantic segmentation


def main(unused_argv):
    print(args)

    np.random.seed(cfg.RNG_SEED)

    # load databases
    print("Setting up image database: " + args.dataset)
    imdb = get_imdb(args.dataset)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)
    print('{:d} roidb entries'.format(len(roidb)))

    if args.dataset_validation != "no":
        print("Setting up validation image database: " + args.dataset_validation)
        imdb_val = get_imdb(args.dataset_validation)
        print('Loaded dataset `{:s}` for validation'.format(imdb_val.name))
        roidb_val = get_training_roidb(imdb_val)
        print('{:d} roidb entries'.format(len(roidb_val)))
    else:
        imdb_val = None
        roidb_val = None


    data_layer = RoIDataLayer(roidb, imdb.num_classes)
    if roidb_val is not None:
        data_layer_val = RoIDataLayer(roidb_val, imdb_val.num_classes, random=True)

    data_layer.forward(1)
    # tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # input and output tensors
    num_classes = len(imdb._classes)

    if "DeepScores" in args.dataset:
        input = tf.placeholder(tf.float32, shape=[None, None, None , 1])
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"

    else:
        input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        resnet_dir = cfg.PRETRAINED_DIR+"/ImageNet/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/VOC2012/"

    # label placeholders
    # dws_labels
    label_dws_energy = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    label_class = tf.placeholder(tf.float32, shape=[None,  None, None, num_classes])
    label_bbox = tf.placeholder(tf.float32, shape=[None, None, None, 2])

    # original bbox label
    label_orig = tf.placeholder(tf.float32, shape=[None, None, 5])

    print("Initializing Model:" + args.model)
    g, init_fn = build_refinenet(input, preset_model = args.model, num_classes=None, pretrained_dir=resnet_dir, substract_mean=False)

    print("Using loss: " + args.loss)
    with tf.variable_scope('deep_watershed'):
        dws_energy = slim.conv2d(g[3], 1, [1, 1], activation_fn=tf.nn.relu, scope='dws_energy')
        energy_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=dws_energy, labels=label_dws_energy))

        dws_mask = tf.squeeze(label_dws_energy >= 0,-1)

        class_logits = slim.conv2d(g[3], num_classes, [1, 1], activation_fn=None, scope='logits')
        class_masked_logits = tf.boolean_mask(class_logits,dws_mask)
        class_masked_labels =  tf.boolean_mask(label_class,dws_mask)
        class_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=class_masked_logits, onehot_labels=class_masked_labels))


        bbox_size = slim.conv2d(g[3], 2, [1, 1], activation_fn=tf.nn.relu, scope='dws_size')
        bbox_masked_predictions = tf.boolean_mask(bbox_size, dws_mask)
        class_masked_labels =  tf.boolean_mask(label_bbox,dws_mask)
        box_loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=bbox_masked_predictions, labels=class_masked_labels))


        ec_loss = tf.add(energy_loss * 1.0, class_loss * 0.5)
        tot_loss = tf.add(ec_loss * 1.0, box_loss * 0.5)


        print("Init optimizers")
        opt_energy = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(energy_loss, var_list=[var for var in
                                                                                                   tf.trainable_variables()])

        opt_ec = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(ec_loss, var_list=[var for var in
                                                                                                   tf.trainable_variables()])

        tot_loss = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(tot_loss, var_list=[var for var in
                                                                                                   tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    # set up saver path
    model_checkpoint_name = cfg.EXP_DIR + "/" + args.dataset+ "/" + args.model
    if args.continue_training or not args.is_training:
        print("Loading checkpoint")
        saver.restore(sess, model_checkpoint_name)
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


    if args.is_training:
        print("Start training")
        for itr in range(1, (args.iterations + 1)):

            # load batch
            blob = data_layer.forward(args.batch_size)

            # from PIL import Image
            # img = Image.fromarray(blob["data"].astype(np.uint8)[0])
            # img.show()
            # img_gt = Image.fromarray((blob["gt_fcn"]+10).astype(np.uint8)*10)
            # img_gt.show()
            if "DeepScores" in args.dataset:
                blob["data"] = np.expand_dims(np.mean(blob["data"], -1), -1)
                #one-hot class labels
                blob["class_map"] = np.eye(imdb.num_classes)[blob["class_map"][:, :, :, -1]]


            # train step
            _, energy_loss_fetch, class_loss_fetch, box_loss_fetch = sess.run([opt_energy, energy_loss,class_loss,box_loss],
                                                            feed_dict={input: blob["data"],
                                                                       label_dws_energy: blob["dws_energy"],
                                                                       label_class: blob["class_map"],
                                                                       label_bbox: blob["bbox_fcn"],
                                                                       label_orig: blob["gt_boxes"] })

            if itr % 7 == 0:
                _, energy_loss_fetch, class_loss_fetch, box_loss_fetch = sess.run(
                    [tot_loss, energy_loss, class_loss, box_loss],
                    feed_dict={input: blob["data"],
                               label_dws_energy: blob["dws_energy"],
                               label_class: blob["class_map"],
                               label_bbox: blob["bbox_fcn"],
                               label_orig: blob["gt_boxes"]})

            if itr == 1:
                print("initial losses:")
                print("energy_loss: " + str(energy_loss_fetch))
                print("class_loss: " + str(class_loss_fetch))
                print("box_loss: " + str(box_loss_fetch))

            if itr % 21 == 0:
                print("loss at itr: " + str(itr))
                print("energy_loss: " + str(energy_loss_fetch))
                print("class_loss: " + str(class_loss_fetch))
                print("box_loss: " + str(box_loss_fetch))

            if itr % 2001 == 0:
                print("saving weights")
                saver.save(sess, model_checkpoint_name)


    else:
        print("Start testing")




def get_training_roidb(imdb):
  """Returns a roidb (Region of Interest database) for use in training."""
  if cfg.TRAIN.USE_FLIPPED:
    print('Appending horizontally-flipped training examples...')
    imdb.append_flipped_images()
    print('done')

  print('Preparing training data...')
  rdl_roidb.prepare_roidb(imdb)
  print('done')

  return imdb.roidb





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
    parser.add_argument("--crop", type=bool, default=True, help="should images be cropped - only applies to DeepScores")
    parser.add_argument("--crop_size", type=bytearray, default=[640,640], help="size of the image to be cropped to - only applies to DeepScores")
    parser.add_argument("--scaling", type=int, default=1,
                        help="scaling factor to be applied before cropping - only applies to DeepScores")
    parser.add_argument("--continue_training", type=bool, default=False, help="load checkpoint")
    parser.add_argument("--iterations", type=int, default=50000, help="nr of batches to train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
    parser.add_argument("--dataset", type=str, default="DeepScores_2017_debug", help="DeepScores, voc or coco")
    parser.add_argument("--dataset_validation", type=str, default="DeepScores_2017_debug", help="DeepScores, voc, coco or no - validation set")
    parser.add_argument("--pretrain_lvl", type=str, default="semseg", help="What kind of pretraining to use: no,class,semseg")
    parser.add_argument("--loss", type=str, default="cross_ent", help="Used loss - cross_ent, regression, bbox")
    parser.add_argument("--is_training", type=bool, default=True, help="Train or validation mode")
    parser.add_argument("--loss_mode", type=str, default="low-dim", help="low-dim or high dim")
    parser.add_argument('--model', type=str, default="RefineNet-Res101", help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152")
    args, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)