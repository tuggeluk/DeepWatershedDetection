import pickle
import numpy as np
import cv2
import os
import re

def build_images_from_pickle():
    list_of_files = os.listdir('/DeepWatershedDetection/visualization/pickle_files')
    for f in list_of_files:
        try:
            with open(os.path.join("/DeepWatershedDetection/visualization/pickle_files", f)) as input_file:
                e = pickle.load(input_file)
            # remove the empty dimensions
            image = np.squeeze(e['data'])
            # boxes = np.squeeze(e['gt_boxes'])
            cv2.imwrite(os.path.join('/DeepWatershedDetection/visualization/images_from_pickle', f[:-6] + 'jpg'), image)
        except KeyError:
            pass


def add_bounding_boxes():
    list_of_files = os.listdir('/DeepWatershedDetection/visualization/images_from_pickle')
    for f in list_of_files:
        img = cv2.imread(os.path.join('/DeepWatershedDetection/visualization/images_from_pickle', f), 1)
        with open(os.path.join("/DeepWatershedDetection/visualization/pickle_files", f[:-3] + 'pickle')) as input_file:
            e = pickle.load(input_file)
        boxes = np.squeeze(e['gt_boxes'])
        for b in boxes:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 1)
        cv2.imwrite(os.path.join('/DeepWatershedDetection/visualization/images_boxes', f), img)

def show_output_images(out_dir, prefixes):
    print("youlo")
    imgs = os.listdir(out_dir)
    nr_imgs = len([x for x in imgs if 'ccomp' in x])
    for i in range(nr_imgs):
        for pref in prefixes:
            im = cv2.imread(out_dir+"/"+pref+str(i)+".png")
            cv2.imshow(pref,im)
        cv2.waitKey()

if __name__ == '__main__':
    out_dir = 'output_images/inference'
    prefixes = ['ccomp', 'gt', 'input', 'stamp_bbox', 'stamp_class', 'stamp_energy']
    show_output_images(out_dir, prefixes)

