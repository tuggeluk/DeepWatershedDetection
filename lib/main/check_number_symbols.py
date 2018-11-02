import numpy as np
import xml.etree.cElementTree as ET
import os
import pandas as pa
import sys


def count_symbols_muscima(classes, set_file):
    # create and initialize the dictionary
    num_syms_per_class = {}
    for k in classes: num_syms_per_class[k] = 0
    xml_annotations_path = '/DeepWatershedDetection/data/MUSICMA++_2017/MUSICMA++_2017/xml_annotations'
    with open(set_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for f in content:
        tree_read = ET.parse(os.path.join(xml_annotations_path, f + '.xml'))
        for o in tree_read.findall('CropObjects'):
            for obj in o.findall('CropObject'):
                obj_name = obj.find('ClassName').text
                if obj_name in num_syms_per_class:
                    num_syms_per_class[obj_name] += 1
    return num_syms_per_class


def count_symbols(classes, set_file):
    # create and initialize the dictionary
    num_syms_per_class = {}
    for k in classes: num_syms_per_class[k] = 0
    xml_annotations_path = '/DeepWatershedDetection/data/MUSICMA++_2017/MUSICMA++_2017/xml_annotations'
    with open(set_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    for f in content:
        tree_read = ET.parse(os.path.join(xml_annotations_path, f + '.xml'))
        for obj in tree_read.findall('object'):
            obj_name = obj.find('name').text
            num_syms_per_class[obj_name] += 1
    return num_syms_per_class


def count_zeros(classes, class_to_ind, dataset):
    num_syms_per_class = count_symbols_muscima(classes, dataset)
    zeros = []
    for k in classes:
        if num_syms_per_class[k] == 0:
            zeros.append(class_to_ind[k])
    return zeros, num_syms_per_class


def count(classes, num_syms_per_class):
    total = 0
    for k in classes:
        total += num_syms_per_class[k]


def main():
    classes = list(pa.read_csv(
        "/DeepWatershedDetection/data/MUSICMA++_2017/MUSICMA_classification/class_names.csv", header=None)[1])
    classes_lower = []
    for k in classes:
        if 'letter' not in k:
            kk = k.lower()
            classes_lower.append(kk)
        else:
            classes_lower.append(k)
    class_to_ind = dict(list(zip(classes_lower, list(range(len(classes))))))
    #class_to_ind = dict(list(zip(classes, list(range(len(classes))))))
    # print(class_to_ind)
    train_set_file = '/DeepWatershedDetection/data/MUSICMA++_2017/train_val_test/train.txt'
    test_set_file = '/DeepWatershedDetection/data/MUSICMA++_2017/train_val_test/test.txt'

    zeros_test, num_syms_per_class_test = count_zeros(classes_lower, class_to_ind, test_set_file)
    zeros_train, num_syms_per_class_train = count_zeros(classes_lower, class_to_ind, train_set_file)
    num_syms_per_class_total = {}
    for k in num_syms_per_class_test:
        num_syms_per_class_total[k] = num_syms_per_class_test[k] + num_syms_per_class_train[k]
    #zeros_test, num_syms_per_class_test = count_zeros(classes, class_to_ind, test_set_file)
    #zeros_train, num_syms_per_class_train = count_zeros(classes, class_to_ind, train_set_file)

    print(num_syms_per_class_test)
    print("\n\n\n\n\n")
    print(num_syms_per_class_train)
    print("\n\n\n\n\n")
    print(num_syms_per_class_total)

    zeros_combined = list(set(zeros_test).union(zeros_train))
    print(zeros_combined)
    list_of_zeros = []
    for k in class_to_ind:
        if class_to_ind[k] in zeros_combined:
            list_of_zeros.append(k)
    print(list_of_zeros)


if __name__ == '__main__':
    main()


