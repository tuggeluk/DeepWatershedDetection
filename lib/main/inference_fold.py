import numpy as np
import os
import cv2
import pickle as cPickle
from PIL import Image
import sys
sys.path.insert(0, '/DeepWatershedDetection/lib')
sys.path.insert(0, os.path.dirname(__file__)[:-4])
import pdb
from datasets.factory import get_imdb
from main.dws_detector import DWSDetector, get_images
from main.config import cfg
import argparse
import time


def main(parsed):
    imdb = get_imdb(parsed.test_set)
    path_model = os.path.join("experiments/music/pretrain_lvl_class/", parsed.net_type, parsed.net_id)
    net = DWSDetector(imdb=imdb, path=path_model, pa=parsed, individual_upsamp=parsed.individual_upsamp)

    path_data = "/home/tugg/Desktop/test_cases"
    path_input = os.path.join(path_data, "input")
    path_output = os.path.join(path_data, "processed")
    inference_folder(net, path_input, path_output)



def inference_folder(net, path_input, path_output):
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


    total_time = []
    images = os.listdir(path_input)
    for i in range(len(images)):
        start_time = time.time()

        if i % 500 == 0:
            print(i)
        if "realistic" not in path:
            im = Image.open(imdb.image_path_at(i)).convert('L')
        else:
            im = Image.open(imdb.image_path_at(i))
        im = np.asanyarray(im)
        im = cv2.resize(im, None, None, fx=parsed.scaling, fy=parsed.scaling, interpolation=cv2.INTER_LINEAR)
        # if im.shape[0]*im.shape[1]>3837*2713:
        #     continue
        if max(im.shape) > 5120:
            print("Image at: "+imdb.image_path_at(i) + "is too large and will be scaled down")
            re_scale = 5120/max(im.shape)
            im = cv2.resize(im, None, None, fx=re_scale, fy=re_scale, interpolation=cv2.INTER_LINEAR)
            final_scale = parsed.scaling*re_scale

        else:
            final_scale = parsed.scaling


        boxes = net.classify_img(im, [7,1], 8)

        if len(boxes) > 1500:
            boxes = []

        for j in range(len(boxes)):
            # invert scaling for Boxes
            boxes[j] = np.array(boxes[j])
            boxes[j][:-1] = (boxes[j][:-1] * (1 / final_scale)).astype(np.int)

            class_of_symbol = boxes[j][4]
            all_boxes[i][class_of_symbol].append(np.array(boxes[j]))

        im = Image.open(imdb.image_path_at(i)).convert('RGB')
        boxes_named =[]
        for box in boxes:
            box = list(box)
            box[-1] = imdb._classes[box[-1]]
            boxes_named.append(box)
        non_annotated, detections = get_images(np.expand_dims(im,0), boxes_named, True, False)
        detections.save(cfg.ROOT_DIR + "/output_images/inference/" + 'prediction' + str(net.counter) + '.png')

        detections.show()
        gt_boxes_named = []
        for row in imdb.o.get_anns(i).iterrows():
            box = row[1]['a_bbox']
            box.append(imdb._classes[imdb._class_ids_to_ind[row[1]['cat_id'][0]]])
            gt_boxes_named.append(box)
        non_annotated, groundtruth = get_images(np.expand_dims(im, 0), gt_boxes_named, True, True)
        groundtruth.save(cfg.ROOT_DIR + "/output_images/inference/" + 'gt' + str(net.counter)+ '.png')
        groundtruth.show()

        end_time = time.time()
        total_time.append(end_time - start_time)





    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=int, default=1, help="scale factor applied to images after loading")

    parser.add_argument("--net_type", type=str, default="RefineNet-Res101", help="type of resnet used (RefineNet-Res152/101)")
    parser.add_argument("--net_id", type=str, default="run_0", help="the id of the net you want to perform inference on")
    parser.add_argument("--dataset", type=str, default='DeepScoresV2', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
    parser.add_argument("--test_set", type=str, default="DeepScoresV2_2020_val", help="dataset to perform inference on")

    parser.add_argument("--saved_net", type=str, default="backbone", help="name (not type) of the net, typically set to backbone")
    parser.add_argument("--energy_loss", type=str, default="softmax", help="type of the energy loss")
    parser.add_argument("--class_loss", type=str, default="softmax", help="type of the class loss")
    parser.add_argument("--bbox_loss", type=str, default="reg", help="type of the bounding boxes loss, must be reg aka regression")
    parser.add_argument("--debug", type=bool, default=False, help="if set to True, it is in debug mode, and instead of running the images on the net, it only evaluates from a previous run")

    parser.add_argument("--individual_upsamp", type=str, default="True", help="is the network built with individual upsamp heads")
    parsed = parser.parse_known_args()
    main(parsed[0])

