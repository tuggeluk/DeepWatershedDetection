# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from PIL import Image

def parse_rec(filename, muscima, rescale_factor=1):
  """ Parse a PASCAL VOC xml file """
  if not muscima:
    tree = ET.parse(filename)
    for size in tree.findall('size'):
        width = int(round(float(size[0].text) * rescale_factor))
        height = int(round(float(size[1].text) * rescale_factor))
    objects = []
    for obj in tree.findall('object'):
      obj_struct = {}
      obj_struct['name'] = obj.find('name').text
      # obj_struct['pose'] = obj.find('pose').text
      bbox = obj.find('bndbox')
      obj_struct['bbox'] = [int(float(bbox.find('xmin').text) * width),
                            int(float(bbox.find('ymin').text) * height),
                            int(float(bbox.find('xmax').text) * width),
                            int(float(bbox.find('ymax').text) * height)]
  else:
    # Parse a muscima xml file
    tree = ET.parse(filename)
    objects = []
    for objs in tree.findall('CropObjects'):
      for obj in objs.findall('CropObject'):
        obj_struct = {}
        obj_struct['name'] = obj.find('ClassName').text
        obj_struct['bbox'] = [int(int(obj.find('Left').text) * rescale_factor),
                              int(int(obj.find('Top').text) * rescale_factor),
                              int((int(obj.find('Left').text) + int(obj.find('Width').text)) * rescale_factor),
                              int((int(obj.find('Top').text) + int(obj.find('Height').text)) * rescale_factor)]
        # we want to ignore the large objects
        if int((int(obj.find('Left').text) + int(obj.find('Width').text)) * rescale_factor) - \
                int(int(obj.find('Left').text) * rescale_factor) < (400 * rescale_factor) and \
                int((int(obj.find('Top').text) + int(obj.find('Height').text)) * rescale_factor) - \
                int(int(obj.find('Top').text) * rescale_factor) < (400 * rescale_factor):
          objects.append(obj_struct)

  return objects


def parse_rec_deepscores(filename, rescale_factor=0.5):
  """ Parse a DeepScores xml file """

  tree = ET.parse(filename)
  objects = []

  # get image size to scale gt bbox
  im_size = Image.open(filename.replace("xml_annotations", "images_png")[:-4] + ".png").convert('L').size
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = "Unspecified"
    obj_struct['truncated'] = int(0)
    obj_struct['difficult'] = int(0)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [(bbox.find('xmin').text),
                          (bbox.find('ymin').text),
                          (bbox.find('xmax').text),
                          (bbox.find('ymax').text)]

    obj_struct['bbox'][0] = int(round(float(obj_struct['bbox'][0]) * im_size[0]))
    obj_struct['bbox'][1] = int(round(float(obj_struct['bbox'][1]) * im_size[1]))
    obj_struct['bbox'][2] = int(round(float(obj_struct['bbox'][2]) * im_size[0]))
    obj_struct['bbox'][3] = int(round(float(obj_struct['bbox'][3]) * im_size[1]))
    objects.append(obj_struct)
    with open('bounding_boxes.pickle', 'w') as f:
      pickle.dump(objects, f)

  return objects


def parse_rec_dota(filename, rescale_factor=0.5):
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        splitlines = [x.strip().split(' ')  for x in lines]
        i = 0
        for splitline in splitlines:
            if i < 2:
              i += 1
              continue
            object_struct = {}
            object_struct['name'] = splitline[8]
            if (len(splitline) == 9):
                object_struct['difficult'] = 0
            elif (len(splitline) == 10):
                object_struct['difficult'] = int(splitline[9])
            # object_struct['difficult'] = 0
            object_struct['bbox'] = [int(float(splitline[0]) * rescale_factor),
                                         int(float(splitline[1]) * rescale_factor),
                                         int(float(splitline[4]) * rescale_factor),
                                         int(float(splitline[5]) * rescale_factor)]
            w = int(float(splitline[4]) * rescale_factor) - int(float(splitline[0]) * rescale_factor)
            h = int(float(splitline[5]) * rescale_factor) - int(float(splitline[1]) * rescale_factor)
            object_struct['area'] = w * h
            objects.append(object_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])
  Top level function that does the PASCAL VOC evaluation.
  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file


  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, '%s_annots.pkl' % imagesetfile)
  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  # remove files not present on disk
  present_files = os.listdir(annopath[:-9])
  present_files = set([x[:-4] for x in present_files])
  imagenames = set(imagenames)
  imagenames = list(imagenames.intersection(present_files))

  if not os.path.isfile(cachefile):
    # load annotations
    recs = {}
    for i, imagename in enumerate(imagenames):
      if "DeepScores" in detpath:
        recs[imagename] = parse_rec_deepscores(annopath.format(imagename))
      elif "MUSICMA" in detpath:
        recs[imagename] = parse_rec(annopath.format(imagename), muscima = True)

      elif "Dota" in detpath:
        recs[imagename] = parse_rec_dota(annopath.format(imagename))

      else:
        recs[imagename] = parse_rec(annopath.format(imagename), musicma = False)
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    with open(cachefile, 'w') as f:
      pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  for imagename in imagenames:
    R = [obj for obj in recs[imagename] if obj['name'] == classname]
    # if len(R) == 0:
    #   continue

    bbox = np.array([x['bbox'] for x in R])


    if len(R) > 0 and "difficult" not in R[0].keys():
      difficult = np.zeros(len(R)).astype(np.bool)
    else:
      difficult = np.array([x['difficult'] for x in R]).astype(np.bool)

    det = [False] * len(R)
    npos = npos + sum(~difficult)
    class_recs[imagename] = {'bbox': bbox,
                             'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  with open(detfile, 'r') as f:
    lines = f.readlines()

  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

  nd = len(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)

  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    for d in range(nd):
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)

      if BBGT.size > 0:
        # compute overlaps
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

      if ovmax > ovthresh:
        if not R['difficult'][jmax]:
          if not R['det'][jmax]:
            tp[d] = 1.
            R['det'][jmax] = 1
          else:
            fp[d] = 1.
      else:
        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap
