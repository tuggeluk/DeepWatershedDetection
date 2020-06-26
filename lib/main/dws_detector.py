from __future__ import print_function
import numpy as np
import tensorflow as tf
from models.dwd_net import build_dwd_net
from main.dws_transform import perform_dws
from PIL import Image
from main.config import cfg
from datasets import fcn_groundtruth


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
        """
        This function classifies an image based on the results of the net, has been tested with different values of cutoff and min_component_size and 
        we have observed that it is very robust to perturbations of those values.
        inputs:
            img - the image, an ndarray
            cutoff - the cutoff we do for the enrgy
            min_component_size - the minimum size of the connected component
        returns:
            dws_list - the list of bounding boxes the dwdnet infers
        """
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
        print(canv.shape)
        print(canv.shape[1]*canv.shape[2])
        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')

        max_input_pix = 2995200
        overlap_pix = 100
        slice_height = int(np.ceil(max_input_pix/canv.shape[2]/160.0))*160
        effective_slice_height = slice_height-overlap_pix
        nr_slices = int(np.ceil(canv.shape[1]/effective_slice_height))

        def clip_append(existing_pred, new_pred, overlap_pix, ind, nr_slices):
            half_overlap = int(overlap_pix/2)
            if ind == 0:
                # only clip end
                new_pred = new_pred[:,:-half_overlap,:,:]
            elif ind == nr_slices-1:
                # only clip start
                new_pred = new_pred[:, half_overlap:-half_overlap, :, :]
            else:
                # clip both
                new_pred = new_pred[:, half_overlap:, :, :]

            if existing_pred is None:
                return new_pred
            else:
                return np.concatenate((existing_pred,new_pred),1)

        pred_energy, pred_class, pred_bbox = [None]*3
        for i in range(nr_slices):
            in_slice = canv[:,effective_slice_height*i:min(effective_slice_height*i+slice_height, canv.shape[1]),:,:]
            slice_pred_energy, slice_pred_class, slice_pred_bbox = self.tf_session.run(
                [self.network_heads["stamp_energy"][self.energy_loss][-1],
                 self.network_heads["stamp_class"][self.class_loss][-1],
                 self.network_heads["stamp_bbox"][self.bbox_loss][-1]], feed_dict={self.input: in_slice})
            pred_energy = clip_append(pred_energy, slice_pred_energy, overlap_pix, i, nr_slices)
            pred_class = clip_append(pred_class, slice_pred_class, overlap_pix, i, nr_slices)
            pred_bbox = clip_append(pred_bbox, slice_pred_bbox, overlap_pix, i, nr_slices)


        save_debug_panes(pred_energy, pred_class, pred_bbox,self.counter)
        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')
        print("forward pass done")
        if self.energy_loss == "softmax":
            pred_energy = np.argmax(pred_energy, axis=3)

        if self.class_loss == "softmax":
            pred_class = np.argmax(pred_class, axis=3)

        if self.bbox_loss == "softmax":
            pred_bbox = np.argmax(pred_bbox, axis=3)


        dws_list = perform_dws(pred_energy, pred_class, pred_bbox, cutoff, min_ccoponent_size)
        #save_images(canv, dws_list, True, False, self.counter)

        self.counter += 1

        return dws_list

    def sliced_forward(self, canv, max_input_pix = 2995200):

        return



def get_images(data, gt_boxes=None, gt=False, text=False):
    """
    Utility function which draws the bounding boxes from both the inference and ground truth, useful to do manual inspection of results
    arguments:
        data - the image in ndarray format.
        boxes - boxes which we want to draw in the image.
        gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
        text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
    returns:
        im_input - the image with the results of the detection.
        im_gt - None if gt is set to False, the image with the drawn ground truth if set to True.
    """
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
    """
    Utility functions which shows the results of get_images in the display window.
    arguments:
        data - the image in ndarray format.
        boxes - boxes which we want to draw in the image.
        gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
        text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
    returns:
        None
    """
    im_input, im_gt = get_images(data, gt_boxes, gt, text)
    im_input.show()
    im_gt.show()


def save_images(data, gt_boxes=None, gt=False, text=False, counter=0):
    """
    Utility function which saves the results of get_images.
    arguments:
        data - the image in ndarray format.
        boxes - boxes which we want to draw in the image.
        gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
        text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
        counter - each image is given a name starting from 0.png, 1.png, ..., num_images.png
    returns:
        None
    """
    im_input, im_gt = get_images(data, gt_boxes, gt, text)

    im_input.save(cfg.ROOT_DIR + "/output_images/inference/" + 'input' + str(counter)+ '.png')
    im_gt.save(cfg.ROOT_DIR + "/output_images/inference/" + 'gt' + str(counter) +'.png')

    return


def save_debug_panes(pred_energy, pred_class, pred_bbox, counter=0):
    """
    Utility function which saves the output panes from the Network directly as images
    arguments:
        pred_energy - energy map prediction
        pred_class - class map prediction
        pred_bbox - bounding box map prediction
        counter - each image is given a name starting from 0.png, 1.png, ..., num_images.png
    returns:
        None
    """
    panes_list = [[pred_energy, "stamp_energy", "softmax"],
    [pred_class, "stamp_class", "softmax"],
    [pred_bbox, "stamp_bbox", "softmax"]]

    for pane in panes_list:
        pan_vis = fcn_groundtruth.color_map(pane[0], {"stamp_func": [pane[1]], "stamp_args": {"loss": pane[2]}})
        im = Image.fromarray(pan_vis[0])
        im.save(cfg.ROOT_DIR + "/output_images/inference/" + pane[1] + str(counter) + '.png')


    return

