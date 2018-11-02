import numpy as np
import os
import cv2
import pickle as cPickle
from PIL import Image
import sys

from datasets.factory import get_imdb
from main.dws_detector import DWSDetector
from main.config import cfg
import argparse
import time

def main(parsed):
    parsed = parsed[0]
    imdb = get_imdb(parsed.test_set)

    path = "/experiments/music_handwritten/pretrain_lvl_semseg/RefineNet-Res101/run_12"
    # path = "/experiments/music/pretrain_lvl_semseg/RefineNet-Res101/run_3"
    # path = "/experiments/realistic/pretrain_lvl_semseg/RefineNet-Res101/run_5"
    debug = False
    if not debug:
        net = DWSDetector(imdb, path)
        all_boxes = test_net(net, imdb, parsed, path)
    else:
        all_boxes = test_net(False, imdb, parsed, path, debug)

def test_net(net, imdb, parsed, path, debug=False):
    """
    This function does inference on the images
    Parameters:
        net - the net we use for the inference
        imdb - the dataset made into compatible form
        parsed - parameters passed from the argparser
        path - the path (string format) for the location of the net
        debug - set it to true if the inference has been already done, and you just want to load the values. Used for debugging purposes.
    """
    output_dir = cfg.OUT_DIR
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    det_file = os.path.join(output_dir, 'detections.pkl')

    print(num_images)

    total_time = []
    if not debug:
        for i in range(num_images):
            start_time = time.time()
            if i%500 == 0:
                print i
            if "realistic" not in path:
                im = Image.open(imdb.image_path_at(i)).convert('L')
            else:
                im = Image.open(imdb.image_path_at(i))
            im = np.asanyarray(im)
            im = cv2.resize(im, None, None, fx=parsed.scaling, fy=parsed.scaling, interpolation=cv2.INTER_LINEAR)
	    print(im.shape)
            if im.shape[0]*im.shape[1]>3837*2713:
	        continue

            boxes = net.classify_img(im,1,4)
	    if len(boxes) > 800:
	        boxes = []
            no_objects = len(boxes)
            for j in range(len(boxes)):
                # invert scaling for Boxes
                boxes[j] = np.array(boxes[j])
                boxes[j][:-1] = (boxes[j][:-1]*(1/parsed.scaling)).astype(np.int)

                class_of_symbol = boxes[j][4]
                all_boxes[class_of_symbol][i].append(np.array(boxes[j]))
            end_time = time.time()
            total_time.append(end_time - start_time)
        print(total_time)
        sum_time = 0
        for t in total_time: sum_time += t
        print(sum_time)
        sys.exit(0)

        # convert to np array
        for i1 in range(len(all_boxes)):
            for i2 in range(len(all_boxes[i1])):
                all_boxes[i1][i2] = np.asarray(all_boxes[i1][i2])

        # inspect all_boxes variable
        with open(det_file, 'wb') as f:
            cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    else:
        with open(det_file, "rb") as f:
            all_boxes = cPickle.load(f)


    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    return all_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--scaling", type=int, default=0.5, help="scale factor applied to images after loading")
    parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_val", help="dataset to perform inference on")


    # configure output heads used ---> have to match trained model
    parsed = parser.parse_known_args()
    main(parsed)
