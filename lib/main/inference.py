import numpy as np
import os
import cv2
import cPickle
from PIL import Image
import sys
sys.path.insert(0, '/DeepWatershedDetection/lib')
sys.path.insert(0,os.path.dirname(__file__)[:-4])
from datasets.factory import get_imdb
from dws_detector import DWSDetector
from config import cfg
import argparse



def main(parsed):
    parsed = parsed[0]
    imdb = get_imdb(parsed.test_set)
    # path = "/experiments/music/pretrain_lvl_semseg/RefineNet-Res101/run_8"
    path = "/experiments/music_handwritten/pretrain_lvl_semseg/RefineNet-Res101/run_0"
    net = DWSDetector(imdb, path)
    all_boxes = test_net(net, imdb, parsed, path)
    #all_boxes = test_net(None, imdb, parsed)


def test_net(net, imdb, parsed, path):
    output_dir = cfg.OUT_DIR
    num_images = len(imdb.image_index)
    # all detections are collected into:
    # all_boxes[cls][image] = N x 5 array of detections in
    # (x1, y1, x2, y2, class)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]
    #output_dir = get_output_dir(imdb, output_dir)
    # timers
    det_file = os.path.join(output_dir, 'detections.pkl')


    print(num_images)
    for i in range(num_images):
        if i%500 == 0:
            print i

        im = Image.open(imdb.image_path_at(i)).convert('L')
        im = np.asanyarray(im)
        im = cv2.resize(im, None, None, fx=parsed.scaling, fy=parsed.scaling, interpolation=cv2.INTER_LINEAR)
        number_of_big_images = 0
        if im.shape[0]*im.shape[1]>3837*2713:
	    number_of_big_images += 1
	    print("Number of big images: " + str(number_of_big_images))
	    continue
	    

        boxes = net.classify_img(im, 5, 4)
	if len(boxes) > 800:
	    boxes = []
        no_objects = len(boxes)
        for j in range(len(boxes)):
            # invert scaling for Boxes
            boxes[j] = np.array(boxes[j])
            boxes[j][:-1] = (boxes[j][:-1]*(1/parsed.scaling)).astype(np.int)

            class_of_symbol = boxes[j][4]
            all_boxes[class_of_symbol][i].append(np.array(boxes[j]))

    # convert to np array
    for i1 in range(len(all_boxes)):
        for i2 in range(len(all_boxes[i1])):
            all_boxes[i1][i2] = np.asarray(all_boxes[i1][i2])

    # inspect all_boxes variable
    with open(det_file, 'wb') as f:
         cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir, path)
    return all_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=int, default=0.5, help="scale factor applied to images after loading")
    parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_test", help="dataset to perform inference on")

    # configure output heads used ---> have to match trained model
    parsed = parser.parse_known_args()
    main(parsed)
