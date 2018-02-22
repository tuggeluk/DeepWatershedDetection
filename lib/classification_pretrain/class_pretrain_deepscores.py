#
# Pretrain resnet 50 / 101 / 152 with deepscores-classification
#
import argparse
from main.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
import pprint
import numpy as np
import deepscores_classification_datareader
from datasets.imdb import imdb
import tensorflow as tf

import os

import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

resnet_v1 = nets.resnet_v1




def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train ResNets on DeepScores-classification')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights if possible',
                      default=True, type=bool)
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
  parser.add_argument('--batch_size', dest='batch_size',
                      help="batchsize",
                      default=200, type=int)
  parser.add_argument('--save_iters', dest='save_iters',
                      help="after how many iterations do we save the model",
                      default=1000, type=int)
  parser.add_argument('--continue', dest='continue_training',
                      help="continue training",
                      default=True, type=bool)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--img_height', dest='img_height',
                      help='set config keys',default=220, type=int)
  parser.add_argument('--img_with', dest='img_with',
                      help='set config keys', default=120, type=int)


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

    img_size = resnet_v1.resnet_v1.default_image_size

    # train set
    print("Setting up image reader...")
    data_reader = deepscores_classification_datareader.class_dataset_reader(cfg.DATA_DIR + "/DeepScores_2017/DeepScores_classification", pad_to=[img_size, img_size])


    imdb = imdb("DeepScores_2017")
    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, args.tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    num_classes = 124

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # Get the selected model.
    # Some of they require pre-trained ResNet
    print("Preparing the model ...")
    input = tf.placeholder(tf.float32, shape=[None, img_size , img_size, 1])
    label = tf.placeholder(tf.int32, shape=[None,num_classes])


    network = None
    init_fn = None

    pretrained_dir = "pretrain_deepscores"
    # load network
    if args.net == 'res50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0005)):
            out, end_points = resnet_v1.resnet_v1_50(input, is_training=args.train, scope='resnet_v1_50', num_classes=num_classes)
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_50.ckpt'), slim.get_model_variables('resnet_v1_50'))
    elif args.net == 'res101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0005)):
            out, end_points = resnet_v1.resnet_v1_101(input, is_training=args.train, scope='resnet_v1_101', num_classes=num_classes)
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_101.ckpt'), slim.get_model_variables('resnet_v1_101'))
    elif args.net == 'res152':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=0.0005)):
            out, end_points = resnet_v1.resnet_v1_152(input, is_training=args.train, scope='resnet_v1_152', num_classes=num_classes)
            # RefineNet requires pre-trained ResNet weights
            init_fn = slim.assign_from_checkpoint_fn(os.path.join(pretrained_dir, 'resnet_v1_152.ckpt'), slim.get_model_variables('resnet_v1_152'))
    else:
    	raise ValueError("Unsupported ResNet model '%s'. This function only supports ResNet 101 and ResNet 152" % (args.net))

    out = tf.squeeze(out,[1,2])
    loss = slim.losses.softmax_cross_entropy(out, label)
    tf.summary.scalar('losses/total_loss', loss)


    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())


    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(label, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)


    # init variables and savers
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # potentioally load weights
    model_checkpoint_name = cfg.PRETRAINED_DIR +"/DeepScores/resnet_v1" + args.net.split("res")[1] + ".ckpt"
    if args.weight:
        if args.continue_training or not args.is_training:
            print('Loaded latest model checkpoint')
            saver.restore(sess, model_checkpoint_name)

    # pre_train_net("asdf", data_reader, output_dir, tb_dir,
    #           pretrained_model=args.weight,
    #           max_iters=args.max_iters)

    # load data into memory
    data_reader.read_images()

    for iter in range(0, args.max_iters):
        batch = data_reader.next_batch(args.batch_size)
        # undo one-hot
        #un_onehot = [np.where(r==1)[0][0] for r in batch[1]]
        train_op.run(session=sess, feed_dict={input: batch[0], label: batch[1]})
        if iter % 10 == 0:
            _, loss_act = sess.run([train_op, loss], feed_dict={input: batch[0], label: batch[1]})
            print(loss_act)

        if iter % args.save_iters == 0:
            save_path = saver.save(sess, model_checkpoint_name)
