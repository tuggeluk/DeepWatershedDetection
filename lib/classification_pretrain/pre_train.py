#
# Pretrain resnet 50 / 101 / 152 with deepscores-classification
#
import argparse
import sys
from main.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import pprint
import numpy as np
import Classification_BatchDataset
import models.resnet_v1 as resnet_v1
from datasets.imdb import imdb
import tensorflow as tf
from tensorflow.contrib import slim
import os


def pre_train_net():
    print("train here")
    return


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights if possible',
                      default=False, type=bool)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--train', dest='train',
                      help="set to train mode true/false",
                      default=True, type=bool)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  # if len(sys.argv) == 1:
  #   parser.print_help()
  #   sys.exit(1)

  args = parser.parse_args()
  return args



if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    print("Setting up image reader...")
    data_reader = Classification_BatchDataset.class_dataset_reader(cfg.DATA_DIR+"/DeepScores_2017/DeepScores_classification")


    imdb = imdb("DeepScores_2017")
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    num_classes = 125

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Get the selected model.
    # Some of they require pre-trained ResNet
    print("Preparing the model ...")
    input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
    label = tf.placeholder(tf.int32, shape=[None, None, None])

    network = None
    init_fn = None

    pretrained_dir = "pretrain_deepscores"
    # load network
    if args.net == 'res50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.TRAIN.WEIGHT_DECAY)):
            out, end_points = resnet_v1.resnet_v1_50(input, is_training=args.train, scope='resnet_v1_50')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_50.ckpt'), slim.get_model_variables('resnet_v1_50'))
    elif args.net == 'res101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.TRAIN.WEIGHT_DECAY)):
            out, end_points = resnet_v1.resnet_v1_101(input, is_training=args.train, scope='resnet_v1_101')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_101.ckpt'), slim.get_model_variables('resnet_v1_101'))
    elif args.net == 'res152':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.TRAIN.WEIGHT_DECAY)):
            out, end_points = resnet_v1.resnet_v1_152(input, is_training=args.train, scope='resnet_v1_152')
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_152.ckpt'), slim.get_model_variables('resnet_v1_152'))
    else:
    	raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 101 and ResNet 152" % (args.net))

    # compute logits
    out = slim.layers.fully_connected(out, 512, scope='fc0')
    out = slim.layers.dropout(out, is_training=args.train)
    logits = slim.layers.fully_connected(out, data_reader.class_names.shape[0], activation_fn=None, scope='fc1')

    # compute loss
    logits = tf.nn.softmax(logits, name='prob')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
    loss = tf.reduce_mean(loss, name='cross_entropy_loss')

    # add optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    # load data into memory
    data_reader.read_images()

    # init variables and savers
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # potentioally load weights
    if args.weight:
        model_checkpoint_name = "checkpoints/classification_" + args.net + "_" + "DeepScores" + ".ckpt"
        if args.continue_training or not args.is_training:
            print('Loaded latest model checkpoint')
            saver.restore(sess, model_checkpoint_name)

    # pre_train_net("asdf", data_reader, output_dir, tb_dir,
    #           pretrained_model=args.weight,
    #           max_iters=args.max_iters)

    for iter in range(0, args.max_iters):
        print("asdf")
