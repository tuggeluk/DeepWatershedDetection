# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ismail Elezi and Lukas Tuggener based on Ross Girshick and Xinlei Chen's code for pascal_voc dataset
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import numpy as np
import scipy.sparse
import pickle
import subprocess
import uuid
from datasets.voc_eval import voc_eval
from main.config import cfg
import math
from obb_anns import OBBAnns
import json


class deep_scoresV2(imdb):
  def __init__(self, image_set, year, devkit_path=None):
    imdb.__init__(self, 'DeepScoresV2' + year + '_' + image_set)
    self._year = year
    self._devkit_path = self._get_default_path() if devkit_path is None \
      else devkit_path

    self._image_set = image_set

    self._data_path = self._devkit_path + "/images"

    self.blacklist = ["staff", 'legerLine']


    self.o = OBBAnns(self._devkit_path+'/deepscores_'+image_set+'.json')
    self.o.load_annotations()
    print(self.o.annotation_sets)
    self.o.set_annotation_set_filter(['deepscores'])
    self.o.set_class_blacklist(self.blacklist)

    self._classes = [v["name"] for (k, v) in self.o.get_cats().items()]
    self._class_ids = [k for (k, v) in self.o.get_cats().items()]

    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._class_ids_to_ind = dict(list(zip(self._class_ids, list(range(self.num_classes)))))
    self._ind_to_class_ids = {v: k for k, v in self._class_ids_to_ind.items()}

    self._image_index = self._load_image_set_index()

    # self.cat_ids = list(self.o.get_cats().keys())
    # self.cat2label = {
    #   cat_id: i
    #   for i, cat_id in enumerate(self.cat_ids)
    # }
    # self.label2cat = {v: k for k, v in self.cat2label.items()}
    # self.CLASSES = tuple([v["name"] for (k, v) in self.o.get_cats().items()])
    # self.img_ids = [id['id'] for id in self.o.img_info]


    self._image_ext = '.png'

    # Default to roidb handler
    self._roidb_handler = self.gt_roidb
    self._salt = str(uuid.uuid4())
    self._comp_id = 'comp4'

    # PASCAL specific config options
    self.config = {'cleanup': True,
                   'use_salt': True,
                   'use_diff': False,
                   'matlab_eval': False,
                   'rpn_file': None}


  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._data_path, self.o.get_imgs(ids=[index])[0]["filename"])
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    image_index = [x["id"] for x in self.o.img_info]
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'DeepScores_' + self._year)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    # if os.path.exists(cache_file):
    #   with open(cache_file, 'rb') as fid:
    #     try:
    #       roidb = pickle.load(fid)
    #     except:
    #       roidb = pickle.load(fid, encoding='bytes')
    #   print('{} gt roidb loaded from {}'.format(self.name, cache_file))
    #   return roidb

    gt_roidb = [self._load_musical_annotation(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2017 or self._image_set != 'debug':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_musical_annotation(self, index):
    """
    Load annotation info from obb_anns in the PASCAL VOC
    format.
    """


    anns = self.o.get_anns(img_id=index)
    boxes = anns['a_bbox']
    boxes = np.round(np.stack(boxes.to_numpy())).astype(np.uint16)

    gt_classes = np.squeeze(np.stack(anns['cat_id'].to_numpy()).astype(np.int32))
    gt_classes = np.array(list(map(self._class_ids_to_ind.get, gt_classes)))
    #blacklisted_anns = [x not in self.blacklist_index for x in gt_classes]
    #boxes = boxes[blacklisted_anns]
    #gt_classes = gt_classes[blacklisted_anns]

    num_objs = boxes.shape[0]
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    for ind in range(boxes.shape[0]):
      seg_areas = (boxes[ind,2]-boxes[ind,0]+1) *(boxes[ind,3]-boxes[ind,1]+1)
      overlaps[ind, gt_classes[ind]] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)
    max(gt_classes)
    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      self._devkit_path,
      'results',
      'musical' + self._year,
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
   for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(str(index), dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output', path=None):
    annopath = os.path.join(
      self._devkit_path,
      'segmentation_detection',
      'xml_annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      self._devkit_path,
      'train_val_test',
      self._image_set + '.txt')
    cachedir = os.path.join(self._devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(self._year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__':
        continue
      filename = self._get_voc_results_file_template().format(cls)
      rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric)
      aps += [ap]
      print(('AP for {} = {:.4f}'.format(cls, ap)))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    # open the file where we want to save the results
    if path is not None:
      res_file = open(os.path.join('/DeepWatershedDetection' + path, 'res.txt'),"w+")
      len_ap = len(aps)
      sum_aps = 0
      present = 0
      for i in range(len_ap):
        print(('{:.3f}'.format(aps[i])))
        if i not in [26, 32,  35, 36, 39, 45, 48, 67, 68, 74, 89, 99, 102, 118]:
          if math.isnan(aps[i]):
            res_file.write(str(0) + "\n")
          else:
            res_file.write(('{:.3f}'.format(aps[i])) + "\n")
            sum_aps += aps[i]
          present += 1
      res_file.write('\n\n\n')
      res_file.write("Mean Average Precision: " + str(sum_aps / float(present)))
      res_file.close()

    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir, path=None):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir, path)
    if self.config['matlab_eval']:
      self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True

  def prepare_json_dict(self, results):
      json_results = {"annotation_set": "deepscores", "proposals": []}
      for idx in range(len(results)):
          img_id = self._image_index[idx]
          result = results[idx]
          for label in range(len(result)):
              bboxes = result[label]
              for i in range(bboxes.shape[0]):
                  data = dict()
                  data['img_id'] = img_id
                  data['bbox'] = [str(nr) for nr in bboxes[i][0:-1]]
                  data['score'] = str(bboxes[i][-1])
                  data['cat_id'] = self._ind_to_class_ids[label]
                  json_results["proposals"].append(data)
      return json_results

  def write_results_json(self, results, filename=None):
      if filename is None:
          filename = "deepscores_results.json"
      json_results = self.prepare_json_dict(results)

      with open(filename, "w") as fo:
          json.dump(json_results, fo)

      return filename

  def evaluate(self,
               results,
               metric='bbox',
               logger=None,
               jsonfile_prefix=None,
               classwise=True,
               proposal_nums=(100, 300, 1000),
               iou_thrs=np.arange(0.5, 0.96, 0.05),
               average_thrs=False,
               store_pickle=True):
      """Evaluation in COCO protocol.

      Args:
          results (list): Testing results of the dataset.
          metric (str | list[str]): Metrics to be evaluated.
          logger (logging.Logger | str | None): Logger used for printing
              related information during evaluation. Default: None.
          jsonfile_prefix (str | None): The prefix of json files. It includes
              the file path and the prefix of filename, e.g., "a/b/prefix".
              If not specified, a temp file will be created. Default: None.
          classwise (bool): Whether to evaluating the AP for each class.
          proposal_nums (Sequence[int]): Proposal number used for evaluating
              recalls, such as recall@100, recall@1000.
              Default: (100, 300, 1000).
          iou_thrs (Sequence[float]): IoU threshold used for evaluating
              recalls. If set to a list, the average recall of all IoUs will
              also be computed. Default: 0.5.

      Returns:
          dict[str: float]
      """

      metrics = metric if isinstance(metric, list) else [metric]
      allowed_metrics = ['bbox']
      for metric in metrics:
          if metric not in allowed_metrics:
              raise KeyError(f'metric {metric} is not supported')

      filename = self.write_results_json(results)

      self.o.load_proposals(filename)
      metric_results = self.o.calculate_metrics(iou_thrs=iou_thrs, classwise=classwise, average_thrs=average_thrs)

      # import pickle
      # with open('evaluation.pickle', 'rb') as input_file:
      #     metric_results = pickle.load(input_file)

      # add Name
      metric_results = {self._classes[self._class_ids_to_ind[key]]: value for (key, value) in metric_results.items()}

      # add occurences
      occurences_by_class = self.o.get_class_occurences()
      for (key, value) in metric_results.items():
          value.update(no_occurences=occurences_by_class[key])

      if store_pickle:
          import pickle
          pickle.dump(metric_results, open('evaluation_renamed.pickle', 'wb'))
      return metric_results


if __name__ == '__main__':
  print("do not execute")
