import numpy as np
import os
import cv2
import cPickle
from PIL import Image
from datasets.factory import get_imdb
from dws_detector import DWSDetector
from config import cfg
import argparse


def main(parsed):
    parsed = parsed[0]
    imdb = get_imdb(parsed.test_set)
    net = DWSDetector(imdb)
    all_boxes = test_net(net, imdb, parsed)
    # all_boxes = test_net(None, imdb, parsed)


def test_net(net, imdb, parsed):
    output_dir = cfg.OUT_DIR
    num_images = len(imdb.image_index)
    # all detections are collected into:
    # all_boxes[cls][image] = N x 5 array of detections in
    # (x1, y1, x2, y2, class)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # output_dir = get_output_dir(imdb, output_dir)

    # timers
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im = Image.open(imdb.image_path_at(i)).convert('L')
        im = np.asanyarray(im)
        im = cv2.resize(im, None, None, fx=parsed.scaling, fy=parsed.scaling, interpolation=cv2.INTER_LINEAR)
        boxes = net.classify_img(im)
        # boxes = np.array([[938, 94, 943, 99, 37], [994, 74, 1006, 85, 29], [994, 74, 1006, 85, 29], [994, 211, 1011, 223, 31]])
        # show_image([np.asanyarray(im)], boxes, True, True)
        no_objects = len(boxes)
        for j in range(len(boxes)):
            class_of_symbol = boxes[j][4]
            all_boxes[class_of_symbol][i].append(np.array(boxes[j]))

    # inspect all_boxes variable
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
    return all_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=int, default=0.5, help="scale factor applied to images after loading")
    parser.add_argument("--test_set", type=str, default="DeepScores_2017_test100", help="dataset to perform inference on")

    # configure output heads used ---> have to match trained model


    parsed = parser.parse_known_args()

    main(parsed)
