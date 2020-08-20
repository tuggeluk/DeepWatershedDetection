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
    parsed = parsed[0]
    imdb = get_imdb(parsed.test_set)
    if parsed.dataset == 'DeepScores':
        path = os.path.join("experiments/music/pretrain_lvl_semseg", parsed.net_type, parsed.net_id)
    elif parsed.dataset == "DeepScoresV2":
        path = os.path.join("experiments/music/pretrain_lvl_class/", parsed.net_type, parsed.net_id)
    elif parsed.dataset == "DeepScores_300dpi":
        path = os.path.join("experiments/music/pretrain_lvl_DeepScores_to_300dpi/", parsed.net_type, parsed.net_id)
    elif parsed.dataset == "MUSCIMA":
        path = os.path.join("experiments/music_handwritten/pretrain_lvl_semseg", parsed.net_type, parsed.net_id)
    elif parsed.dataset == "Dota":
        path = os.path.join("experiments/realistic/pretrain_lvl_semseg", parsed.net_type, parsed.net_id)
    elif parsed.dataset == "VOC":
        path = os.path.join("experiments/realistic/pretrain_lvl_class", parsed.net_type, parsed.net_id)
    if not parsed.debug:
        net = DWSDetector(imdb=imdb, path=path, pa=parsed, individual_upsamp=parsed.individual_upsamp)

        all_boxes = test_net(net, imdb, parsed, path)
    else:
        all_boxes = test_net(False, imdb, parsed, path, parsed.debug)


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
    all_boxes = [[[] for _ in range(imdb.num_classes)]
                 for _ in range(num_images)]

    det_file = os.path.join(output_dir, 'detections.pkl')

    print(num_images)

    total_time = []
    if not debug:
        for i in range(num_images):

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
        print(total_time)
        sum_time = 0
        for t in total_time: sum_time += t
        print(sum_time)

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
    imdb.evaluate(all_boxes)
    #imdb.evaluate_detections(all_boxes, output_dir, path)
    return all_boxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=int, default=1, help="scale factor applied to images after loading")
    dataset = 'DeepScoresV2'
    if dataset == 'MUSCIMA':
        parser.add_argument("--dataset", type=str, default='MUSCIMA', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
        parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_test", help="dataset to perform inference on")
    elif dataset == 'DeepScoresV2':
        parser.add_argument("--dataset", type=str, default='DeepScoresV2', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
        parser.add_argument("--test_set", type=str, default="DeepScoresV2_2020_val", help="dataset to perform inference on")
    elif dataset == 'DeepScores':
        parser.add_argument("--dataset", type=str, default='DeepScores', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
        parser.add_argument("--test_set", type=str, default="DeepScores_2017_test", help="dataset to perform inference on")
    elif dataset == 'DeepScores_300dpi':
        parser.add_argument("--dataset", type=str, default='DeepScores_300dpi', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
        parser.add_argument("--test_set", type=str, default="DeepScores_300dpi_2017_val", help="dataset to perform inference on, we use val for evaluation, test can be used only visually")
    elif dataset == 'Dota':
        parser.add_argument("--dataset", type=str, default='Dota', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
        parser.add_argument("--test_set", type=str, default="Dota_2018_debug", help="dataset to perform inference on")
    elif dataset == 'VOC':
        parser.add_argument("--dataset", type=str, default='VOC', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota, VOC")
        parser.add_argument("--test_set", type=str, default="voc_2012_train", help="dataset to perform inference on, voc_2012_val/voc_2012_train")
    parser.add_argument("--net_type", type=str, default="RefineNet-Res101", help="type of resnet used (RefineNet-Res152/101)")
    parser.add_argument("--net_id", type=str, default="run_0", help="the id of the net you want to perform inference on")

    parser.add_argument("--saved_net", type=str, default="backbone", help="name (not type) of the net, typically set to backbone")
    parser.add_argument("--energy_loss", type=str, default="softmax", help="type of the energy loss")
    parser.add_argument("--class_loss", type=str, default="softmax", help="type of the class loss")
    parser.add_argument("--bbox_loss", type=str, default="reg", help="type of the bounding boxes loss, must be reg aka regression")
    parser.add_argument("--debug", type=bool, default=False, help="if set to True, it is in debug mode, and instead of running the images on the net, it only evaluates from a previous run")

    parser.add_argument("--individual_upsamp", type=str, default="True", help="is the network built with individual upsamp heads")

    parsed = parser.parse_known_args()
    main(parsed)

