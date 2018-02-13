# pretrain fully-conv networks on semantic segmentation tasks
# training-code adapted from https://github.com/DrSleep/tensorflow-deeplab-resnet

import tensorflow as tf
import numpy as np
import argparse
import sys
import pascalvoc_semseg_datareader
import deepscores_semseg_datareader
from main.config import cfg
from models.RefineNet import build_refinenet





def main(unused_argv):
    print("Setting up image reader...")
    if args.dataset == "DeepScores":
        data_reader = deepscores_semseg_datareader.ds_semseg_datareader(cfg.DATA_DIR+"/DeepScores_2017/segmentation_detection", crop=args.crop, crop_size=args.crop_size)
        resnet_dir = cfg.PRETRAINED_DIR+"/DeepScores/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/DeepScores_semseg/"
        num_classes = 124
        input = tf.placeholder(tf.float32, shape=[None, args.crop_size[0], args.crop_size[1], 1])
        substract_mean = False
    elif args.dataset == "VOC2012":
        data_reader = pascalvoc_semseg_datareader.voc_seg_dataset_reader(cfg.DATA_DIR+"/VOC2012")
        resnet_dir = cfg.PRETRAINED_DIR+"/ImageNet/"
        refinenet_dir = cfg.PRETRAINED_DIR+"/VOC2012/"
        num_classes = 21
        input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        substract_mean = True
    else:
        print("Unknown dataset")
        sys.exit(1)

    #train_images, train_annotations = data_reader.next_batch(20)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # init model
    print("Preparing the model ...")
    output = tf.placeholder(tf.float32, shape=[None, None, None, num_classes])

    network, init_fn = build_refinenet(input, preset_model = args.model, num_classes=num_classes,pretrained_dir=resnet_dir, substract_mean=substract_mean)

    # init optimizer
    # Compute your (unweighted) softmax cross entropy loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=output))

    opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in
                                                                                               tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1000)
    sess.run(tf.global_variables_initializer())

    # load classification weights or continue training from older checkpoint
    # If a pre-trained ResNet is required, load the weights. --> depends on path tiven to build_refinenet
    # This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
    print("loading resnet-weights")
    if init_fn is not None:
        init_fn(sess)


    # Load a previous checkpoint if desired
    model_checkpoint_name = refinenet_dir + args.model + ".ckpt"
    if args.continue_training:
        print('Loaded latest model checkpoint')
        saver.restore(sess, model_checkpoint_name)

    print("start training")
    # Train for a fixed number of batches
    for itr in range(1,(args.iterations+1)):

        train_images, train_annotations = data_reader.next_batch(args.batch_size)
        # VOC only works with batchsize 1
        if args.dataset == "VOC2012":
            train_images = np.expand_dims(train_images[0],0)
            train_annotations = np.expand_dims(train_annotations[0],0)

        # for i in range(0,train_images.shape[0]):
        #     train_annotations[i] = np.eye(num_classes)[train_annotations[i][:,:,-1]]

        # convert annotation to one-hot enc
        train_annotations = np.eye(num_classes)[train_annotations[:,:,:,-1]]

        # size test
        # h = 400
        # w = 500
        # train_images = np.zeros((1,h,w,3))
        # train_annotations = np.zeros((1,h,w,21))
        # h and w has to be a multiple of 32!!
        # gets taken care of by data loaders

        _,current = sess.run([opt,loss], feed_dict={input: train_images, output: train_annotations})

        if itr == 1:
            print("initial loss" + str(current))

        if itr % 21 == 0:
            print("loss of current batch:"+str(current))

        if itr % 2001 == 0:
            print("saving weights")
            saver.save(sess, model_checkpoint_name)

    print("Compute test error")
    test_images, test_annotations = data_reader.get_test_records()
    test_annotations = np.eye(num_classes)[test_annotations[:, :, :, -1]]
    test_losses = []

    for itr in range(0,len(test_images)/args.batch_size):
        test_img_b = test_images[(itr*args.batch_size):((itr+1)*args.batch_size)]
        test_ann_b = test_annotations[(itr * args.batch_size):((itr + 1) * args.batch_size)]
        cur_loss = sess.run([loss], feed_dict={input: test_img_b, output: test_ann_b})
        test_losses.append(cur_loss)

    print("Test Loss:")
    print(np.mean(test_losses))



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
  parser.add_argument("--crop", type=bool, default=True, help="should images be cropped")
  parser.add_argument("--continue_training", type=bool, default=False, help="load checkpoint")
  parser.add_argument("--crop_size", type=bytearray, default=[640,640], help="batch size for training")
  parser.add_argument("--iterations", type=int, default=50000, help="path to logs directory")
  parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
  parser.add_argument("--dataset", type=str, default="DeepScores", help="DeepScores or VOC2012")
  parser.add_argument('--model', type=str, default="RefineNet-Res101", help='The model you are using. Currently supports: RefineNet-Res50, RefineNet-Res101, RefineNet-Res152')
  args, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)