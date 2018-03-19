from __future__ import print_function
import numpy as np
import tensorflow as tf
from models.dwd_net import build_dwd_net
from dws_transform import perform_dws
from PIL import Image

np.random.seed(314)
tf.set_random_seed(314)


class DWSDetector:
    def __init__(self, imdb):
        self.model_path = "trained_models/RefineNet-Res101"
        self.model_name = "RefineNet-Res101"
        self.saved_net = 'backbone'
        self.tf_session = None
        self.root_dir = '/home/revan/PycharmProjects/DeepWatershedDetection/'
        self.sess = tf.Session()
        print('Loading model')
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 1])
        self.network_heads, self.init_fn = build_dwd_net(self.input, model=self.model_name, num_classes=imdb.num_classes,
                                               pretrained_dir="", substract_mean=False)
        self.saver = tf.train.Saver(max_to_keep=1000)
        self.sess.run(tf.global_variables_initializer())
        print("Loading weights")
        self.saver.restore(self.sess, self.root_dir + "/" + self.model_path + "/" + self.saved_net)
        self.tf_session = self.sess

    def classify_img(self, img):
        if len(img.shape) < 4:
            img = np.expand_dims(np.expand_dims(img, -1), 0)

        y_mulity = int(np.ceil(img.shape[1] / 320.0))
        x_mulity = int(np.ceil(img.shape[2] / 320.0))
        canv = np.ones([y_mulity * 320, x_mulity * 320], dtype=np.uint8) * 255
        canv = np.expand_dims(np.expand_dims(canv, -1), 0)

        canv[0, 0:img.shape[1], 0:img.shape[2]] = img[0]
        pred_energy, pred_class_logits, pred_bbox = self.tf_session.run(
            [self.network_heads["stamp_energy"]["reg"][0], self.network_heads["stamp_class"]["softmax"][0],
             self.network_heads["stamp_bbox"]["reg"][0]], feed_dict={self.input: canv})
        pred_class = np.argmax(pred_class_logits, axis=3)

        dws_list = perform_dws(pred_energy, pred_class, pred_bbox)

        return dws_list


def show_image(data, gt_boxes=None, gt=False, text=False):
    from PIL import ImageDraw
    im = Image.fromarray(data[0].astype("uint8"))
    im.show()

    if gt:
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes:
            draw.rectangle(((row[0],row[1]),(row[2],row[3])), fill="red")
        im.show()
    if text:
        draw = ImageDraw.Draw(im)
        # overlay GT boxes
        for row in gt_boxes:
            draw.text((row[2],row[3]),row[4], fill="red")
        im.show()

    return