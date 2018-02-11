
import os
import tensorflow as tf
import numpy as np
import argparse
import sys
from main.config import cfg
from models.RefineNet import build_refinenet
from datasets.factory import get_imdb
import roi_data_layer.roidb as rdl_roidb


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

    # load database
    print("Setting up image database: " + args.dataset)
    imdb = get_imdb(args.dataset)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)
    print('{:d} roidb entries'.format(len(roidb)))

    # determ


    print("Initializing Model:" + args.model)


    print("Loading Checkpoint for: " + args.pretrain_lvl +" pre training")


    print("Using loss: " + args.loss)


    print("Init optimizer")


    print("Start training")


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
    parser.add_argument("--pretrain_lvl", type=str, default="semseg", help="What kind of pretraining to use: no,class,semseg")
    parser.add_argument("--loss", type=str, default="cross_ent", help="Used loss - cross_ent, regression, bbox")
    parser.add_argument("--is_training", type=bool, default=True, help="Train or validation mode")
    parser.add_argument('--model', type=str, default="RefineNet-Res101", help="Base model -  Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152")
    args, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)