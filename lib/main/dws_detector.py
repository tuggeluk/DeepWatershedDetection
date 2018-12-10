from __future__ import print_function
import numpy as np
import tensorflow as tf
from models.dwd_net import build_dwd_net
from main.dws_transform import perform_dws
from PIL import Image
from main.config import cfg
from datasets import fcn_groundtruth
import sys
import cv2
import pdb

np.random.seed(314)
tf.set_random_seed(314)


class DWSDetector:
    def __init__(self, imdb, path, pa, individual_upsamp = False):
        self.model_path = path
        self.model_name = pa.net_type
        self.saved_net = pa.saved_net
        # has to be adjusted according to the training scheme used
        self.energy_loss = pa.energy_loss
        self.class_loss = pa.class_loss
        self.bbox_loss = pa.bbox_loss

        self.tf_session = None
        self.root_dir = cfg.ROOT_DIR
        self.sess = tf.Session()
        print('Loading model')

        if "realistic" in self.model_path:
            self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        else:
            self.input = tf.placeholder(tf.float32, shape=[None, None, None, 1])

        self.network_heads, self.init_fn = build_dwd_net(self.input, model=self.model_name, num_classes=imdb.num_classes,
                                               pretrained_dir="", substract_mean=False,  individual_upsamp = individual_upsamp)

        self.saver = tf.train.Saver(max_to_keep=1000)
        self.sess.run(tf.global_variables_initializer())
        print("Loading weights")
        self.saver.restore(self.sess, self.root_dir + "/" + self.model_path + "/" + self.saved_net)
        self.tf_session = self.sess
        self.counter = 0

    def classify_img(self, img, cutoff=0, min_ccoponent_size=0):

        if img.shape[0] > 1:
            img = np.expand_dims(img, 0)

        if img.shape[-1] > 3:
            img = np.expand_dims(img, -1)

        y_mulity = int(np.ceil(img.shape[1] / 160.0))
        x_mulity = int(np.ceil(img.shape[2] / 160.0))
        if "realistic" not in self.model_path:
            canv = np.ones([y_mulity * 160, x_mulity * 160], dtype=np.uint8) * 255
            canv = np.expand_dims(np.expand_dims(canv, -1), 0)
        else:
            canv = np.ones([y_mulity * 160, x_mulity * 160, 3], dtype=np.uint8) * 255
            canv = np.expand_dims(canv, 0)

        canv[0, 0:img.shape[1], 0:img.shape[2]] = img[0]

        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')

        pred_energy, pred_class, pred_bbox = self.tf_session.run(
            [self.network_heads["stamp_energy"][self.energy_loss][-1],
             self.network_heads["stamp_class"][self.class_loss][-1],
             self.network_heads["stamp_bbox"][self.bbox_loss][-1]], feed_dict={self.input: canv})

        save_debug_panes(canv, pred_energy, pred_class, pred_bbox,self.counter)

        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')
        if self.energy_loss == "softmax":
            pred_energy = np.argmax(pred_energy, axis=3)

        if self.class_loss == "softmax":
            pred_class = np.argmax(pred_class, axis=3)

        if self.bbox_loss == "softmax":
            pred_bbox = np.argmax(pred_bbox, axis=3)


        dws_list = perform_dws(pred_energy, pred_class, pred_bbox, cutoff, min_ccoponent_size)
        save_images(canv, dws_list, True, False, self.counter)

        self.counter += 1


        return dws_list


def get_images(data, gt_boxes=None, gt=False, text=False):
    if data.shape[-1] == 1:
        data = data.squeeze(-1)

    from PIL import ImageDraw
    im_input = Image.fromarray(data[0].astype("uint8"))
    im_gt = None

    if gt:
        im_gt = Image.fromarray(data[0].astype("uint8"))
        draw = ImageDraw.Draw(im_gt)
        # overlay GT boxes
        for row in gt_boxes:
            # cv2.rectangle(im_input, (row[0], row[1], row[2], row[3]), (0, 255, 0), 1)
            draw.rectangle(((row[0], row[1]), (row[2], row[3])), fill="red")

    if text:
        draw = ImageDraw.Draw(im_gt)
        # overlay GT boxes
        for row in gt_boxes:
            draw.text((row[2], row[3]), str(row[4]), fill="red")

    return im_input, im_gt


def show_images(data, gt_boxes=None, gt=False, text=False):
    im_input, im_gt = get_images(data, gt_boxes, gt, text)
    im_input.show()
    im_gt.show()

    return


def save_images(data, gt_boxes=None, gt=False, text=False, counter=0):
    im_input, im_gt = get_images(data, gt_boxes, gt, text)
    im_input.save(cfg.ROOT_DIR + "/output_images/inference/" +  'input' + str(counter)+ '.png')
    im_gt.save(cfg.ROOT_DIR + "/output_images/inference/"  + 'gt' + str(counter) +'.png')

    return

def save_debug_panes(data, pred_energy, pred_class, pred_bbox, counter=0):
    # save panes to disk
    print("save panes to disk")
    panes_list = [[pred_energy, "stamp_energy", "softmax"],
    [pred_class, "stamp_class", "softmax"],
    [pred_bbox, "stamp_bbox", "softmax"]]

    for pane in panes_list:
        pan_vis = fcn_groundtruth.color_map(pane[0], {"stamp_func" : [pane[1]], "stamp_args": {"loss": pane[2]}})
        im = Image.fromarray(pan_vis[0])
        im.save(cfg.ROOT_DIR + "/output_images/inference/" + pane[1] + str(counter) + '.png')


    return
