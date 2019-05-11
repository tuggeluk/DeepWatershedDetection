import numpy as np
import os
import cv2
import pickle
from PIL import Image
import sys
sys.path.insert(0, '/DeepWatershedDetection/lib')
sys.path.insert(0, os.path.dirname(__file__)[:-4])
import pdb
from datasets.factory import get_imdb
from main.dws_detector import DWSDetector
#from main.config import cfg
import time
import datetime


def main(parsed, model_dir , do_debug= False):

    imdb = get_imdb(parsed,parsed.test_set)
    if not do_debug:
        net = DWSDetector(parsed, model_dir, imdb)

        all_boxes = test_net(net, imdb, parsed, model_dir)
    else:
        all_boxes = test_net(False, parsed, do_debug)


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
    output_dir = ""
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
                print(i)
            if "DeepScores" in path:
                im = Image.open(imdb.image_path_at(i)).convert('L')
            else:
                if parsed.paired_data > 1:
                    # hacky only for macrophages
                    im_1 = Image.open(imdb.image_path_at(i))
                    im_1 = np.array(im_1, dtype=np.float32) / 256

                    im_2 = Image.open(imdb.image_path_at(i).replace("DAPI", "mCherry"))
                    im_2 = np.array(im_2, dtype=np.float32) / 256

                    im = np.stack([im_1, im_2, np.zeros(im_1.shape)], -1)

                else:
                    im = Image.open(imdb.image_path_at(i))

            im = np.asanyarray(im)
            im = cv2.resize(im, None, None, fx=parsed.scale_list[0], fy=parsed.scale_list[0], interpolation=cv2.INTER_LINEAR)
            if im.shape[0]*im.shape[1]>3837*2713:
                continue

            boxes = net.classify_img(im, 1, 4)
            if len(boxes) > 800:
                boxes = []
            no_objects = len(boxes)
            for j in range(len(boxes)):
                # invert scaling for Boxes
                boxes[j] = np.array(boxes[j])
                boxes[j][:-1] = (boxes[j][:-1] * (1 / parsed.scaling)).astype(np.int)

                class_of_symbol = boxes[j][4]
                all_boxes[class_of_symbol][i].append(np.array(boxes[j]))
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
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    else:
        pdb.set_trace()
        with open(det_file, "rb") as f:
            all_boxes = pickle.load(f)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir, path)
    return all_boxes


if __name__ == '__main__':

    # test
    # config_files = [['2019-03-06T09:15:03.387376', '6b534eeae099fd5e5ba35e9e78a3ca9d3f56af43139f7998d6721b42.p'],
    #                 ['2019-04-06T09:15:03.387376', '6b534eeae099fd5e5ba35e9e78a3ca9d3f56af43139f7998d6721b42.p'],
    #                 ['2018-05-06T09:15:03.387376', '6b534eeae099fd5e5ba35e9e78a3ca9d3f56af43139f7998d6721b42.p'],
    #                 ['2019-05-06T09:15:03.387376', '6b534eeae099fd5e5ba35e9e78a3ca9d3f56af43139f7998d6721b42.p'],
    #                 ['2019-05-06T09:30:03.387376', '6b534eeae099fd5e5ba35e9e78a3ca9d3f56af43139f7998d6721b42.p']
    #                 ]

    trained_model_dir = "/share/DeepWatershedDetection/experiments/macrophages/pretrain_lvl_class/RefineNet-Res101/run_3"

    # get latest config
    now = datetime.datetime.now()
    config_files = [x.split("__") for x in os.listdir(trained_model_dir) if x[-2:] == '.p']

    for ix, ele in enumerate(config_files):
        ele.append(now-datetime.datetime.strptime(ele[0], '%Y-%m-%dT%H:%M:%S.%f'))

    config_files.sort(key=lambda x: x[2])

    with open(trained_model_dir+"/"+config_files[0][0]+"__"+config_files[0][1], "rb") as f:
        config = pickle.load(f)


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--scaling", type=int, default=1, help="scale factor applied to images after loading")
    # dataset = 'DeepScores'
    # if dataset == 'MUSCIMA':
    #     parser.add_argument("--dataset", type=str, default='MUSCIMA', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
    #     parser.add_argument("--test_set", type=str, default="MUSICMA++_2017_test", help="dataset to perform inference on")
    # elif dataset == 'DeepScores':
    #     parser.add_argument("--dataset", type=str, default='DeepScores', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
    #     parser.add_argument("--test_set", type=str, default="DeepScores_2017_test", help="dataset to perform inference on")
    # elif dataset == 'DeepScores_300dpi':
    #     parser.add_argument("--dataset", type=str, default='DeepScores_300dpi', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
    #     parser.add_argument("--test_set", type=str, default="DeepScores_300dpi_2017_val", help="dataset to perform inference on, we use val for evaluation, test can be used only visually")
    # elif dataset == 'Dota':
    #     parser.add_argument("--dataset", type=str, default='Dota', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota")
    #     parser.add_argument("--test_set", type=str, default="Dota_2018_debug", help="dataset to perform inference on")
    # elif dataset == 'VOC':
    #     parser.add_argument("--dataset", type=str, default='VOC', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota, VOC")
    #     parser.add_argument("--test_set", type=str, default="voc_2012_train", help="dataset to perform inference on, voc_2012_val/voc_2012_train")
    # elif dataset == 'macrophages':
    #     parser.add_argument("--dataset", type=str, default='macrophages_2019_train', help="name of the dataset: DeepScores, DeepScores_300dpi, MUSCIMA, Dota, VOC")
    #     parser.add_argument("--test_set", type=str, default="macrophages_2019_test", help="dataset to perform inference on, voc_2012_val/voc_2012_train")
    #
    # parser.add_argument("--net_type", type=str, default="RefineNet-Res152", help="type of network used")
    # parser.add_argument("--net_id", type=str, default="run_0", help="the id of the net you want to perform inference on")
    #
    # parser.add_argument("--saved_net", type=str, default="backbone", help="name (not type) of the net, typically set to backbone")
    # parser.add_argument("--energy_loss", type=str, default="softmax", help="type of the energy loss")
    # parser.add_argument("--class_loss", type=str, default="softmax", help="type of the class loss")
    # parser.add_argument("--bbox_loss", type=str, default="reg", help="type of the bounding boxes loss, must be reg aka regression")
    # parser.add_argument("--debug", type=bool, default=False, help="if set to True, it is in debug mode, and instead of running the images on the net, it only evaluates from a previous run")
    #
    #
    # parsed = parser.parse_known_args()

    main(config,trained_model_dir)

