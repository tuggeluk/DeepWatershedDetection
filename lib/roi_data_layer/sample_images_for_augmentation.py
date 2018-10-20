import os
import random
import pickle
import xml.etree.ElementTree as ET
import pandas as pa
import sys


class RandomImageSampler:
    def __init__(self, height, width):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.pickle_relative_directory = 'new_augmentation_combined'
        self.pickle_absolute_directory_path, self.absolute_xml_path = None, None
        self.list_of_images, self.list_of_files = [], []
        self.bounding_boxes = []
        self.classes = list(pa.read_csv(
            "/DeepWatershedDetection/data/DeepScores_2017/DeepScores_classification/class_names.csv",
            header=None)[1])
        self.num_classes = 124
        self.class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
        self.height = height
        self.width = width
        self.small_width = 80
        self.small_height = 130

    def get_paths(self):
        self.r_path = self.root_path.split('/')
        self.pickle_absolute_directory_path = os.path.join('/', self.r_path[1], self.pickle_relative_directory, 'pickle_files')
        self.absolute_xml_path = os.path.join('/', self.r_path[1], self.pickle_relative_directory, 'xml_data_gt')
        return self.pickle_absolute_directory_path, self.absolute_xml_path

    def sample_image_up(self, ignore_symbols=1, vertical=0):
        """ A much more efficient function of sampling small images
            inputs: ignore_symbols - if 1 it ignores everything bar the symbol in the center """
        self.pickle_absolute_directory_path, self.absolute_xml_path = self.get_paths()
        # get the list of all pickle files
        pickle_files = os.listdir(self.pickle_absolute_directory_path)
        # choose one of the pickle files randomly
        p_file = random.choice(pickle_files)
        with open(os.path.join(self.pickle_absolute_directory_path, p_file), 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            num_images = self.check_augment()
            for i in range(num_images):
                key = random.choice(info.keys())
                while key == 'timeSig16' or key == 'timeSig12':
                    key = random.choice(info.keys())
                key_num = self.class_to_ind[key]
                im = random.choice(info[key])
                self.list_of_images.append(im[0])  # images
                if ignore_symbols:
                    for box in im[2]:
                        if box[-1] == key_num:
                            self.bounding_boxes.append(box)  # bounding boxes - consider only the symbol in the middle
                            break
                else:
                    self.bounding_boxes.append(im[2])
        return self.list_of_images, self.bounding_boxes, num_images, self.small_height, self.small_width

    def sample_image_full(self, ignore_symbols=1, horizontal=12, vertical=8):
        self.pickle_absolute_directory_path, self.absolute_xml_path = self.get_paths()
        # get the list of all pickle files
        pickle_files = os.listdir(self.pickle_absolute_directory_path)
        # choose one of the pickle files randomly
        p_file = random.choice(pickle_files)
        with open(os.path.join(self.pickle_absolute_directory_path, p_file), 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            for i in range(horizontal):
                for j in range(vertical):
                    key = random.choice(info.keys())
                    while key == 'timeSig16' or key == 'timeSig12':
                        key = random.choice(info.keys())
                    key_num = self.class_to_ind[key]
                    im = random.choice(info[key])
                    self.list_of_images.append(im[0])  # images
                    if ignore_symbols:
                        for box in im[2]:
                            if box[-1] == key_num:
                                self.bounding_boxes.append(box)  # bounding boxes - consider only the symbol in the middle
                                break
                    else:
                        self.bounding_boxes.append(im[2])
        return self.list_of_images, self.bounding_boxes, horizontal, vertical, self.small_height, self.small_width

    def sample_image_old(self):
        """ A less efficient function of sampling small images, it needs to be used if you're using xml files for ground_truth
            instead of lists from a pickle file. The function is deprecated and will be removed for the final version of the code """
        self.pickle_absolute_directory_path, self.absolute_xml_path = self.get_paths()
        # get the list of all pickle files
        pickle_files = os.listdir(self.pickle_absolute_directory_path)
        # choose one of the pickle files randomly
        p_file = random.choice(pickle_files)
        with open(os.path.join(self.pickle_absolute_directory_path, p_file), 'rb') as pickle_file:
            info = pickle.load(pickle_file)
            num_images = self.check_augment()
            for i in range(num_images):
                key = random.choice(info.keys())
                if key != 'timeSig16' and key != 'timeSig12':
                    im = random.choice(info[key])
                    self.list_of_images.append(im[0])  # images
                    self.list_of_files.append(im[2])  # xml_paths
        self.list_of_files = self.fix_xml_path()
        for f in self.list_of_files:
            self.bounding_boxes.append(self.parse_rec(f))
        return self.list_of_images, self.bounding_boxes, num_images, self.small_height, self.small_width

    def fix_xml_path(self):
        fixed_name_xml_files = []
        for f in self.list_of_files:
            f_name_list = f.split('/')
            f_name = os.path.join(self.absolute_xml_path, f_name_list[-2], f_name_list[-1])
            fixed_name_xml_files.append(f_name)
        return fixed_name_xml_files

    def parse_rec(self, filename):
        """ Parse an xml file """
        tree = ET.parse(filename)
        for size in tree.findall('size'):
            width = int(round(float(size[0].text)))
            height = int(round(float(size[1].text)))
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(float(bbox.find('xmin').text) * width * 0.5),
                                  int(float(bbox.find('ymin').text) * height * 0.5),
                                  int(float(bbox.find('xmax').text) * width * 0.5),
                                  int(float(bbox.find('ymax').text) * height * 0.5),
                                  self.class_to_ind[obj_struct['name']]]
            objects.append(obj_struct['bbox'])
        return objects

    def check_augment(self):
        # check if we can augment on vertical axis
        if self.height % self.small_height == 0:
            num_images = self.height / self.small_height
            self.where = 'width'
        # otherwise augment on horizontal axis
        else:
            num_images = self.width / self.small_width
            self.where = 'height'
        return num_images  # , self.where
